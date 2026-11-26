package com.iris.localai.video

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.graphics.Bitmap
import com.iris.localai.onnx.OnnxBundle
import com.iris.localai.onnx.OnnxBundleLibrary
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.CancellationException
import kotlin.math.max
import kotlin.math.min
import kotlin.random.Random

/** Strict Stable Video Diffusion ONNX adapter: image encoder, VAE encoder, 5-D UNet and VAE decoder. */
object VideoOnnxEngine {
    const val WIDTH = 512
    const val HEIGHT = 288
    const val FRAME_COUNT = 14
    const val FPS = 6
    private const val LATENT_CHANNELS = 4
    private const val LATENT_SCALE = 0.18215f

    fun generateMp4(
        filesDir: File,
        bundle: OnnxBundle,
        firstFrame: Bitmap,
        output: File,
        motionBucketId: Int = 127,
        seed: Int? = null,
        onProgress: (Int) -> Unit = {},
        isCancelled: () -> Boolean = { Thread.currentThread().isInterrupted },
    ): File {
        require(firstFrame.width > 0 && firstFrame.height > 0) { "Selected image is empty" }
        require(motionBucketId in 0..255) { "Motion bucket must be between 0 and 255" }
        val imageEncoderFile = OnnxBundleLibrary.resolve(filesDir, bundle, "image_encoder.onnx")
        val vaeEncoderFile = OnnxBundleLibrary.resolve(filesDir, bundle, "vae_encoder.onnx")
        val unetFile = OnnxBundleLibrary.resolve(filesDir, bundle, "unet.onnx")
        val vaeDecoderFile = OnnxBundleLibrary.resolve(filesDir, bundle, "vae_decoder.onnx")
        val schedulerFile = OnnxBundleLibrary.resolve(filesDir, bundle, "scheduler_config.json")
        val scheduler = SvdEulerScheduler.fromJson(schedulerFile.readText())
        val environment = OrtEnvironment.getEnvironment()
        val imageOptions = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        val vaeEncoderOptions = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        val unetOptions = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        val vaeDecoderOptions = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        val random = seed?.let { Random(it) } ?: Random.Default
        try {
            environment.createSession(imageEncoderFile.absolutePath, imageOptions).use { imageSession ->
                environment.createSession(vaeEncoderFile.absolutePath, vaeEncoderOptions).use { encoderSession ->
                    environment.createSession(unetFile.absolutePath, unetOptions).use { unetSession ->
                        environment.createSession(vaeDecoderFile.absolutePath, vaeDecoderOptions).use { decoderSession ->
                            val clipPixels = imageTensorData(firstFrame, 224, 224, clipNormalize = true)
                            val imageEmbedding = runSingleInput(
                                environment, imageSession, "pixel_values", clipPixels, longArrayOf(1, 3, 224, 224),
                            ).values
                            val vaePixels = imageTensorData(firstFrame, WIDTH, HEIGHT, clipNormalize = false)
                            val encodedLatent = runSingleInput(
                                environment, encoderSession, "sample", vaePixels, longArrayOf(1, 3, HEIGHT.toLong(), WIDTH.toLong()),
                                outputNames = setOf("latent_sample", "sample", "latents"),
                            )
                            require(encodedLatent.shape.size == 4 && encodedLatent.shape[1] == LATENT_CHANNELS.toLong()) {
                                "VAE encoder output must have shape [1,4,height/8,width/8]"
                            }
                            val latentHeight = encodedLatent.shape[2].toInt()
                            val latentWidth = encodedLatent.shape[3].toInt()
                            require(latentHeight > 0 && latentWidth > 0) { "VAE encoder returned invalid latent dimensions" }
                            val expectedSpatialSize = latentHeight * latentWidth
                            val imageLatent = encodedLatent.values
                            require(imageLatent.size == LATENT_CHANNELS * expectedSpatialSize) {
                                "VAE encoder output has an unsupported batch shape"
                            }
                            val timeIds = floatArrayOf((FPS - 1).toFloat(), motionBucketId.toFloat(), 0f)
                            val totalLatents = FRAME_COUNT * LATENT_CHANNELS * expectedSpatialSize
                            var latents = FloatArray(totalLatents) { random.nextGaussian().toFloat() * scheduler.initialSigma }
                            val condition = FloatArray(totalLatents)
                            for (frame in 0 until FRAME_COUNT) {
                                for (index in imageLatent.indices) {
                                    condition[frame * imageLatent.size + index] = imageLatent[index] * LATENT_SCALE
                                }
                            }

                            for (step in 0 until scheduler.steps) {
                                checkCancelled(isCancelled)
                                val modelInput = buildUnetInput(
                                    scheduler.scaleModelInput(latents, step),
                                    condition,
                                    FRAME_COUNT,
                                    expectedSpatialSize,
                                )
                                val predicted = runUnet(
                                    environment,
                                    unetSession,
                                    modelInput,
                                    imageEmbedding,
                                    timeIds,
                                    scheduler.timesteps[step],
                                    latentHeight,
                                    latentWidth,
                                )
                                val guided = FloatArray(latents.size)
                                val batchSize = latents.size
                                require(predicted.size == batchSize * 2) {
                                    "SVD UNet output must contain unconditional and conditional batches"
                                }
                                for (index in latents.indices) {
                                    val unconditional = predicted[index]
                                    val conditionalPrediction = predicted[batchSize + index]
                                    guided[index] = unconditional + 3f * (conditionalPrediction - unconditional)
                                }
                                latents = scheduler.step(guided, step, latents)
                                onProgress(((step + 1) * 70 / scheduler.steps).coerceIn(1, 70))
                            }

                            val decoded = ArrayList<Bitmap>(FRAME_COUNT)
                            try {
                                for (frame in 0 until FRAME_COUNT) {
                                    checkCancelled(isCancelled)
                                    val frameLatent = FloatArray(LATENT_CHANNELS * expectedSpatialSize) { index ->
                                        latents[frame * LATENT_CHANNELS * expectedSpatialSize + index] / LATENT_SCALE
                                    }
                                    val pixels = runSingleInput(
                                        environment,
                                        decoderSession,
                                        "latent_sample",
                                        frameLatent,
                                        longArrayOf(1, LATENT_CHANNELS.toLong(), latentHeight.toLong(), latentWidth.toLong()),
                                        outputNames = setOf("sample", "image", "pixel_values"),
                                        inputNames = setOf("latent_sample", "sample", "latents"),
                                    )
                                    decoded += pixelsToBitmap(pixels.values, pixels.shape)
                                    onProgress(70 + (frame + 1) * 20 / FRAME_COUNT)
                                }
                                checkCancelled(isCancelled)
                                if (output.exists()) throw IOException("Output file already exists: ${output.name}")
                                Mp4VideoWriter.write(output, decoded, FPS, onProgress = { fraction ->
                                    onProgress(90 + fraction.coerceIn(0, 10))
                                })
                            } finally {
                                decoded.forEach { it.recycle() }
                            }
                            return output
                        }
                    }
                }
            }
        } finally {
            imageOptions.close()
            vaeEncoderOptions.close()
            unetOptions.close()
            vaeDecoderOptions.close()
        }
    }

    private data class FloatOutput(val values: FloatArray, val shape: LongArray)

    private fun runSingleInput(
        environment: OrtEnvironment,
        session: OrtSession,
        inputName: String,
        values: FloatArray,
        shape: LongArray,
        outputNames: Set<String> = emptySet(),
        inputNames: Set<String> = setOf(inputName),
    ): FloatOutput {
        val actualInput = session.inputInfo.keys.singleOrNull()
            ?: throw IllegalArgumentException("Expected a single-input ONNX graph")
        require(actualInput in inputNames) {
            "Expected input ${inputNames.joinToString(" or ")}; found $actualInput"
        }
        val tensor = OnnxTensor.createTensor(environment, directFloatBuffer(values), shape)
        try {
            return session.run(mapOf(actualInput to tensor)).use { result ->
                val names = result.map { it.key }.toList()
                val outputName = outputNames.firstOrNull { it in names }
                    ?: names.firstOrNull { result.get(it).orElse(null) is OnnxTensor }
                    ?: throw IllegalStateException("ONNX graph returned no tensor output")
                result.floatOutput(outputName)
            }
        } finally {
            tensor.close()
        }
    }

    private fun imageTensorData(bitmap: Bitmap, width: Int, height: Int, clipNormalize: Boolean): FloatArray {
        val scaled = centerCrop(bitmap, width, height)
        try {
            val pixels = IntArray(width * height)
            scaled.getPixels(pixels, 0, width, 0, 0, width, height)
            val output = FloatArray(3 * width * height)
            val plane = width * height
            for (i in pixels.indices) {
                val pixel = pixels[i]
                val red = ((pixel shr 16) and 255) / 255f
                val green = ((pixel shr 8) and 255) / 255f
                val blue = (pixel and 255) / 255f
                if (clipNormalize) {
                    output[i] = (red - 0.48145466f) / 0.26862954f
                    output[plane + i] = (green - 0.4578275f) / 0.26130258f
                    output[plane * 2 + i] = (blue - 0.40821073f) / 0.27577711f
                } else {
                    output[i] = red * 2f - 1f
                    output[plane + i] = green * 2f - 1f
                    output[plane * 2 + i] = blue * 2f - 1f
                }
            }
            return output
        } finally {
            if (scaled !== bitmap) scaled.recycle()
        }
    }

    private fun centerCrop(source: Bitmap, width: Int, height: Int): Bitmap {
        val scale = max(width.toFloat() / source.width, height.toFloat() / source.height)
        val scaledWidth = (source.width * scale).toInt().coerceAtLeast(width)
        val scaledHeight = (source.height * scale).toInt().coerceAtLeast(height)
        val scaled = Bitmap.createScaledBitmap(source, scaledWidth, scaledHeight, true)
        val left = (scaledWidth - width) / 2
        val top = (scaledHeight - height) / 2
        val cropped = Bitmap.createBitmap(scaled, left, top, width, height)
        if (cropped !== scaled) scaled.recycle()
        return cropped
    }

    private fun buildUnetInput(
        scaledLatents: FloatArray,
        imageCondition: FloatArray,
        frames: Int,
        spatialSize: Int,
    ): FloatArray {
        val perBatch = frames * 8 * spatialSize
        val output = FloatArray(perBatch * 2)
        for (batch in 0..1) {
            for (frame in 0 until frames) {
                val sourceOffset = frame * LATENT_CHANNELS * spatialSize
                val targetOffset = batch * perBatch + frame * 8 * spatialSize
                for (channel in 0 until LATENT_CHANNELS) {
                    val targetChannelOffset = targetOffset + channel * spatialSize
                    val conditionChannelOffset = targetOffset + (channel + LATENT_CHANNELS) * spatialSize
                    for (pixel in 0 until spatialSize) {
                        output[targetChannelOffset + pixel] = scaledLatents[sourceOffset + channel * spatialSize + pixel]
                        output[conditionChannelOffset + pixel] = imageCondition[sourceOffset + channel * spatialSize + pixel]
                    }
                }
            }
        }
        return output
    }

    private fun runUnet(
        environment: OrtEnvironment,
        session: OrtSession,
        sample: FloatArray,
        imageEmbedding: FloatArray,
        timeIds: FloatArray,
        timestep: Float,
        latentHeight: Int,
        latentWidth: Int,
    ): FloatArray {
        val expectedSampleShape = longArrayOf(2, FRAME_COUNT.toLong(), 8, latentHeight.toLong(), latentWidth.toLong())
        val inputs = LinkedHashMap<String, OnnxTensor>()
        try {
            session.inputInfo.keys.forEach { name ->
                inputs[name] = when (name) {
                    "sample" -> OnnxTensor.createTensor(environment, directFloatBuffer(sample), expectedSampleShape)
                    "timestep" -> {
                        val shape = findInputShape(session, name)
                        val dimensions = shape.map { if (it < 0) 1L else it }.toLongArray()
                        require(dimensions.isEmpty() || dimensions.fold(1L) { total, dimension -> total * dimension } == 1L) {
                            "SVD timestep input must be a scalar"
                        }
                        OnnxTensor.createTensor(environment, directFloatBuffer(floatArrayOf(timestep)), dimensions)
                    }
                    "encoder_hidden_states" -> {
                        val shape = findInputShape(session, name)
                        val batchStride = imageEmbedding.size
                        val doubled = FloatArray(batchStride * 2)
                        imageEmbedding.copyInto(doubled, batchStride)
                        OnnxTensor.createTensor(environment, directFloatBuffer(doubled), shapeForEmbeddings(shape, imageEmbedding.size))
                    }
                    "added_time_ids" -> OnnxTensor.createTensor(
                        environment,
                        directFloatBuffer(timeIds + timeIds),
                        longArrayOf(2, 3),
                    )
                    else -> throw IllegalArgumentException("Unsupported SVD UNet input: $name")
                }
            }
            require("sample" in inputs && "encoder_hidden_states" in inputs && "added_time_ids" in inputs) {
                "SVD UNet must accept sample, encoder_hidden_states and added_time_ids"
            }
            return session.run(inputs).use { result ->
                val outputName = result.firstTensorName(setOf("out_sample", "sample"))
                result.floatOutput(outputName).values
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    private fun shapeForEmbeddings(metadataShape: LongArray, size: Int): LongArray {
        require(metadataShape.size == 3 && metadataShape[0] in setOf(-1L, 2L)) {
            "SVD encoder_hidden_states must have [batch,sequence,hidden] shape"
        }
        val shape = metadataShape.copyOf()
        shape[0] = 2
        require(shape.drop(1).all { it > 0 } && shape[1] * shape[2] == size.toLong()) {
            "SVD image embedding output does not match the UNet input"
        }
        return shape
    }

    private fun findInputShape(session: OrtSession, name: String): LongArray =
        (session.inputInfo[name]?.info as? TensorInfo)?.shape
            ?: throw IllegalArgumentException("$name is not a tensor input")

    private fun directFloatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder())
            .asFloatBuffer().apply { put(values); flip() }

    private fun pixelsToBitmap(values: FloatArray, shape: LongArray): Bitmap {
        require(shape.size == 4 && shape[0] == 1L && shape[1] == 3L) {
            "VAE decoder output must have shape [1,3,height,width]"
        }
        val height = shape[2].toInt()
        val width = shape[3].toInt()
        require(values.size == 3 * width * height) { "VAE decoder returned an invalid image tensor" }
        val pixels = IntArray(width * height)
        val plane = width * height
        for (i in pixels.indices) {
            fun channel(index: Int): Int = (((values[index] + 1f) * 127.5f).toInt()).coerceIn(0, 255)
            val r = channel(i)
            val g = channel(plane + i)
            val b = channel(plane * 2 + i)
            pixels[i] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun OrtSession.Result.floatOutput(name: String): FloatOutput {
        val tensor = get(name).orElseThrow { IllegalStateException("ONNX output not found: $name") } as? OnnxTensor
            ?: throw IllegalStateException("ONNX output $name is not a tensor")
        val info = tensor.info as? TensorInfo ?: throw IllegalStateException("ONNX output $name has no tensor metadata")
        val buffer = tensor.floatBuffer
        val values = FloatArray(buffer.remaining())
        buffer.get(values)
        return FloatOutput(values, info.shape)
    }

    private fun OrtSession.Result.firstTensorName(preferred: Set<String>): String {
        val names = map { it.key }.toList()
        return preferred.firstOrNull { it in names }
            ?: names.firstOrNull { get(it).orElse(null) is OnnxTensor }
            ?: throw IllegalStateException("ONNX graph returned no tensor outputs")
    }

    private fun checkCancelled(isCancelled: () -> Boolean) {
        if (isCancelled()) throw CancellationException("Video generation cancelled")
    }
}

/** EulerDiscreteScheduler with the scaled-linear beta schedule used by common SVD checkpoints. */
private class SvdEulerScheduler(
    trainSteps: Int,
    betaStart: Double,
    betaEnd: Double,
    val steps: Int = 25,
) {
    val sigmas: FloatArray
    val timesteps: FloatArray
    val initialSigma: Float

    init {
        require(trainSteps >= steps && steps in 1..100)
        val start = kotlin.math.sqrt(betaStart)
        val end = kotlin.math.sqrt(betaEnd)
        val alphas = DoubleArray(trainSteps)
        var product = 1.0
        for (i in 0 until trainSteps) {
            val value = start + (end - start) * i / (trainSteps - 1)
            product *= 1.0 - value * value
            alphas[i] = product
        }
        val trainSigmas = DoubleArray(trainSteps) { kotlin.math.sqrt((1.0 - alphas[it]) / alphas[it]) }
        sigmas = FloatArray(steps + 1)
        timesteps = FloatArray(steps)
        for (i in 0 until steps) {
            val t = (trainSteps - 1).toDouble() * (steps - 1 - i) / max(steps - 1, 1)
            val low = t.toInt()
            val high = min(low + 1, trainSteps - 1)
            val fraction = t - low
            timesteps[i] = t.toFloat()
            sigmas[i] = (trainSigmas[low] * (1.0 - fraction) + trainSigmas[high] * fraction).toFloat()
        }
        sigmas[steps] = 0f
        initialSigma = sigmas[0]
    }

    fun scaleModelInput(sample: FloatArray, step: Int): FloatArray {
        val denominator = kotlin.math.sqrt(sigmas[step] * sigmas[step] + 1f)
        return FloatArray(sample.size) { sample[it] / denominator }
    }

    fun step(modelOutput: FloatArray, step: Int, sample: FloatArray): FloatArray {
        require(modelOutput.size == sample.size)
        val delta = sigmas[step + 1] - sigmas[step]
        return FloatArray(sample.size) { sample[it] + modelOutput[it] * delta }
    }

    companion object {
        fun fromJson(json: String): SvdEulerScheduler {
            val root = Json.parseToJsonElement(json).jsonObject
            val steps = root["num_train_timesteps"]?.jsonPrimitive?.intOrNull ?: 1000
            val betaStart = root["beta_start"]?.jsonPrimitive?.doubleOrNull ?: 0.00085
            val betaEnd = root["beta_end"]?.jsonPrimitive?.doubleOrNull ?: 0.012
            val betaSchedule = root["beta_schedule"]?.jsonPrimitive?.contentOrNull
            require(betaSchedule == null || betaSchedule == "scaled_linear") {
                "Only the SVD scaled_linear Euler scheduler is supported"
            }
            val predictionType = root["prediction_type"]?.jsonPrimitive?.contentOrNull
            require(predictionType == null || predictionType == "epsilon") {
                "Only epsilon-prediction SVD ONNX models are supported"
            }
            return SvdEulerScheduler(steps, betaStart, betaEnd)
        }
    }
}

private fun Random.nextGaussian(): Double {
    val first = nextDouble().coerceAtLeast(1e-12)
    val second = nextDouble()
    return kotlin.math.sqrt(-2.0 * kotlin.math.ln(first)) * kotlin.math.cos(2.0 * Math.PI * second)
}
