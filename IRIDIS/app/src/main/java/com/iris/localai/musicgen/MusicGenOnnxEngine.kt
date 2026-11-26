package com.iris.localai.musicgen

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import com.iris.localai.onnx.OnnxBundle
import com.iris.localai.onnx.OnnxBundleLibrary
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import kotlin.math.max
import kotlin.random.Random

/** Inference adapter for the explicitly supported MusicGen merged-decoder ONNX bundle. */
object MusicGenOnnxEngine {
    data class Output(val wav: File, val sampleRate: Int, val samples: Int)

    fun generate(
        filesDir: File,
        bundle: OnnxBundle,
        prompt: String,
        output: File,
        durationSeconds: Int = 5,
        seed: Int? = null,
        isCancelled: () -> Boolean = { Thread.currentThread().isInterrupted },
    ): Output {
        require(prompt.isNotBlank()) { "Enter a music prompt" }
        require(durationSeconds in 1..20) { "Duration must be between 1 and 20 seconds" }
        checkCancelled(isCancelled)
        val configFile = OnnxBundleLibrary.resolve(filesDir, bundle, "config.json")
        val tokenizerFile = OnnxBundleLibrary.resolve(filesDir, bundle, "tokenizer.json")
        val textFile = OnnxBundleLibrary.resolve(filesDir, bundle, "text_encoder.onnx")
        val decoderFile = OnnxBundleLibrary.resolve(filesDir, bundle, "decoder_model_merged.onnx")
        val codecFile = OnnxBundleLibrary.resolve(filesDir, bundle, "encodec_decode.onnx")
        val config = Json.parseToJsonElement(configFile.readText()).jsonObject
        val codebooks = config["num_codebooks"]?.jsonPrimitive?.intOrNull
            ?: config["audio_encoder"]?.jsonObject?.get("num_codebooks")?.jsonPrimitive?.intOrNull
            ?: 4
        require(codebooks == 4) { "This adapter currently supports MusicGen models with four codebooks" }
        val vocabSize = config["audio_encoder"]?.jsonObject?.get("codebook_size")?.jsonPrimitive?.intOrNull
            ?: config["decoder"]?.jsonObject?.get("vocab_size")?.jsonPrimitive?.intOrNull
            ?: 2048
        val frameRate = config["audio_encoder"]?.jsonObject?.get("frame_rate")?.jsonPrimitive?.intOrNull
            ?: config["frame_rate"]?.jsonPrimitive?.intOrNull
            ?: 50
        require(vocabSize in 2..65_536 && frameRate in 1..200) { "Unsupported MusicGen configuration" }
        val stepCount = durationSeconds * frameRate
        val tokenizer = UnigramTokenizer.fromTokenizerJson(tokenizerFile.readText())
        val textIds = tokenizer.encode(prompt)
        require(textIds.isNotEmpty() && textIds.size <= 1024) { "Prompt is too long for the text encoder" }
        val environment = OrtEnvironment.getEnvironment()
        val random = seed?.let { Random(it) } ?: Random.Default
        val textOptions = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        val decoderOptions = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        val codecOptions = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        try {
            environment.createSession(textFile.absolutePath, textOptions).use { textSession ->
                environment.createSession(decoderFile.absolutePath, decoderOptions).use { decoderSession ->
                    environment.createSession(codecFile.absolutePath, codecOptions).use { codecSession ->
                        val hidden = encodeText(environment, textSession, textIds, tokenizer.padTokenId())
                        val tokenRows = ArrayList<IntArray>(stepCount + codebooks - 1)
                        var previous = IntArray(codebooks) { vocabSize }
                        var past = emptyMap<String, FloatCache>()
                        val cacheInputs = decoderSession.inputInfo.keys.filter { it.startsWith("past_key_values") }
                        require(cacheInputs.isNotEmpty()) {
                            "Merged MusicGen decoder must expose past_key_values inputs for cached autoregressive inference"
                        }
                        for (step in 0 until stepCount + codebooks - 1) {
                            checkCancelled(isCancelled)
                            val inputs = makeDecoderInputs(
                                environment = environment,
                                session = decoderSession,
                                previous = previous,
                                hidden = hidden,
                                cache = past,
                                cacheInputs = cacheInputs,
                                useCache = step > 0,
                            )
                            val result = try {
                                decoderSession.run(inputs)
                            } finally {
                                inputs.values.forEach { it.close() }
                            }
                            result.use { outputs ->
                                val logits = outputs.floatOutput("logits")
                                val vocabPerBook = logits.shape.lastOrNull()?.toInt()
                                    ?.takeIf { it == vocabSize }
                                    ?: if (logits.values.size % (codebooks * vocabSize) == 0) vocabSize else -1
                                require(vocabPerBook == vocabSize && logits.values.size >= codebooks * vocabSize) {
                                    "Decoder logits must contain $codebooks codebooks of $vocabSize tokens"
                                }
                                val offset = logits.values.size - codebooks * vocabSize
                                previous = IntArray(codebooks) { book ->
                                    sampleTopK(
                                        logits.values,
                                        offset + book * vocabSize,
                                        vocabSize,
                                        topK = 250,
                                        temperature = 1.0,
                                        random = random,
                                    )
                                }
                                tokenRows += previous.copyOf()
                                past = readPresentCache(outputs, cacheInputs)
                                require(past.size == cacheInputs.size) {
                                    "Merged MusicGen decoder did not return every present_key_values cache"
                                }
                            }
                        }

                        val audioCodes = IntArray(codebooks * stepCount)
                        for (book in 0 until codebooks) {
                            for (frame in 0 until stepCount) {
                                audioCodes[book * stepCount + frame] = tokenRows[frame + book][book]
                            }
                        }
                        checkCancelled(isCancelled)
                        val pcm = decodeAudio(environment, codecSession, audioCodes, stepCount, codebooks)
                        if (output.exists()) throw IOException("Output file already exists: ${output.name}")
                        PcmWavWriter.writeMono16(output, pcm, 32_000)
                        return Output(output, 32_000, pcm.size)
                    }
                }
            }
        } finally {
            textOptions.close()
            decoderOptions.close()
            codecOptions.close()
        }
    }

    private data class HiddenState(val values: FloatArray, val shape: LongArray, val attentionMask: LongArray)
    private data class FloatCache(val values: FloatArray, val shape: LongArray)
    private data class FloatOutput(val values: FloatArray, val shape: LongArray)

    private fun encodeText(
        environment: OrtEnvironment,
        session: OrtSession,
        ids: LongArray,
        padTokenId: Long,
    ): HiddenState {
        val inputShape = tensorInfo(session, "input_ids").shape
        require(inputShape.size == 2) { "Text encoder input_ids must have rank 2" }
        val sequenceLength = inputShape[1].takeIf { it > 0 }?.toInt() ?: ids.size
        require(ids.size <= sequenceLength) { "Prompt exceeds the text encoder's fixed input length" }
        val paddedIds = LongArray(sequenceLength) { if (it < ids.size) ids[it] else padTokenId }
        val attentionMask = LongArray(sequenceLength) { if (it < ids.size) 1L else 0L }
        val inputs = LinkedHashMap<String, OnnxTensor>()
        try {
            session.inputInfo.keys.forEach { name ->
                val info = tensorInfo(session, name)
                when (name) {
                    "input_ids" -> inputs[name] = longTensor(environment, paddedIds, longArrayOf(1, sequenceLength.toLong()))
                    "attention_mask", "encoder_attention_mask" ->
                        inputs[name] = longTensor(environment, attentionMask, longArrayOf(1, sequenceLength.toLong()))
                    else -> throw IllegalArgumentException("Unsupported text encoder input: $name")
                }
                require(info.shape.size == 2) { "Text encoder input $name must have rank 2" }
            }
            require("input_ids" in inputs) { "Text encoder is missing input_ids" }
            return session.run(inputs).use { result ->
                val name = result.firstTensorName(setOf("last_hidden_state", "text_embeds", "hidden_states"))
                result.floatOutput(name).let { HiddenState(it.values, it.shape, attentionMask) }
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    private fun makeDecoderInputs(
        environment: OrtEnvironment,
        session: OrtSession,
        previous: IntArray,
        hidden: HiddenState,
        cache: Map<String, FloatCache>,
        cacheInputs: List<String>,
        useCache: Boolean,
    ): LinkedHashMap<String, OnnxTensor> {
        val inputs = LinkedHashMap<String, OnnxTensor>()
        try {
            session.inputInfo.keys.forEach { name ->
                when {
                    name == "input_ids" || name == "decoder_input_ids" -> {
                        val shape = tokenShape(tensorInfo(session, name).shape, previous.size)
                        val ids = LongArray(previous.size) { previous[it].toLong() }
                        inputs[name] = longTensor(environment, ids, shape)
                    }
                    name == "encoder_hidden_states" ->
                        inputs[name] = floatTensor(environment, hidden.values, hidden.shape)
                    name == "encoder_attention_mask" || name == "attention_mask" ->
                        inputs[name] = longTensor(environment, hidden.attentionMask, longArrayOf(1, hidden.attentionMask.size.toLong()))
                    name == "use_cache_branch" -> {
                        val shape = tensorInfo(session, name).shape.map { if (it < 0) 1L else it }.toLongArray()
                        require(shape.isEmpty() || shape.fold(1L) { total, dimension -> total * dimension } == 1L) {
                            "use_cache_branch must be a scalar boolean"
                        }
                        val value = ByteBuffer.allocateDirect(1)
                            .put((if (useCache) 1 else 0).toByte())
                            .apply { flip() }
                        inputs[name] = OnnxTensor.createTensor(environment, value, shape, OnnxJavaType.BOOL)
                    }
                    name.startsWith("past_key_values") -> {
                        val value = cache[name] ?: if (!useCache) emptyCache(tensorInfo(session, name).shape) else
                            throw IllegalStateException("Missing decoder cache input: $name")
                        inputs[name] = floatTensor(environment, value.values, value.shape)
                    }
                    else -> throw IllegalArgumentException("Unsupported merged decoder input: $name")
                }
            }
            require(inputs.keys.any { it == "input_ids" || it == "decoder_input_ids" }) {
                "Merged decoder is missing input_ids"
            }
            require("encoder_hidden_states" in inputs) { "Merged decoder is missing encoder_hidden_states" }
            require(cacheInputs.all { it in inputs }) { "Not all decoder cache inputs were bound" }
            return inputs
        } catch (e: Exception) {
            inputs.values.forEach { it.close() }
            throw e
        }
    }

    private fun readPresentCache(result: OrtSession.Result, cacheInputs: List<String>): Map<String, FloatCache> {
        val names = result.map { it.key }.toSet()
        val values = LinkedHashMap<String, FloatCache>()
        cacheInputs.forEach { inputName ->
            val suffix = cacheSuffix(inputName)
            val outputName = names.firstOrNull { name ->
                (name.startsWith("present") || name.startsWith("present_key_values")) &&
                    cacheSuffix(name) == suffix
            } ?: names.firstOrNull { name ->
                (name.startsWith("present") || name.startsWith("present_key_values")) &&
                    name.substringAfterLast('.') == inputName.substringAfterLast('.')
            } ?: throw IllegalStateException("Merged decoder output is missing cache for $inputName")
            val output = result.floatOutput(outputName)
            values[inputName] = FloatCache(output.values, output.shape)
        }
        return values
    }

    private fun decodeAudio(
        environment: OrtEnvironment,
        session: OrtSession,
        codes: IntArray,
        steps: Int,
        codebooks: Int,
    ): FloatArray {
        val inputs = LinkedHashMap<String, OnnxTensor>()
        try {
            session.inputInfo.keys.forEach { name ->
                require(name == "audio_codes") { "Unsupported EnCodec decoder input: $name" }
                val ids = LongArray(codes.size) { codes[it].toLong() }
                inputs[name] = longTensor(environment, ids, longArrayOf(1, 1, codebooks.toLong(), steps.toLong()))
            }
            require("audio_codes" in inputs) { "EnCodec decoder is missing audio_codes" }
            return session.run(inputs).use { result ->
                val outputName = result.firstTensorName(setOf("audio_values", "waveform", "audio"))
                val output = result.floatOutput(outputName)
                require(output.shape.isNotEmpty() && output.shape.last() > 0 &&
                    output.shape.dropLast(1).all { it == 1L } &&
                    output.values.size.toLong() == output.shape.last()
                ) { "EnCodec decoder must return one mono audio waveform" }
                output.values
            }
        } finally {
            inputs.values.forEach { it.close() }
        }
    }

    private fun sampleTopK(
        logits: FloatArray,
        offset: Int,
        vocabSize: Int,
        topK: Int,
        temperature: Double,
        random: Random,
    ): Int {
        val candidates = (0 until vocabSize).asSequence()
            .filter { logits[offset + it].isFinite() }
            .sortedByDescending { logits[offset + it] }
            .take(topK)
            .toList()
        require(candidates.isNotEmpty()) { "Decoder returned no finite logits" }
        val maxLogit = logits[offset + candidates.first()]
        val weights = DoubleArray(candidates.size) { index ->
            kotlin.math.exp(((logits[offset + candidates[index]] - maxLogit) / temperature).toDouble())
        }
        val total = weights.sum()
        var choice = random.nextDouble() * total
        for (index in weights.indices) {
            choice -= weights[index]
            if (choice <= 0.0) return candidates[index]
        }
        return candidates.last()
    }

    private fun emptyCache(shape: LongArray): FloatCache {
        require(shape.size == 4) { "Only rank-4 MusicGen attention cache tensors are supported" }
        val resolved = shape.mapIndexed { index, dimension ->
            when {
                dimension > 0 -> dimension
                index == 0 -> 1L
                index == 2 -> 0L
                else -> throw IllegalArgumentException("Cannot infer decoder cache input shape")
            }
        }.toLongArray()
        return FloatCache(FloatArray(0), resolved)
    }

    private fun tokenShape(metadataShape: LongArray, tokenCount: Int): LongArray {
        require(metadataShape.size in 2..3) { "Unsupported decoder input_ids rank: ${metadataShape.size}" }
        val resolved = metadataShape.copyOf()
        resolved.indices.forEach { i ->
            if (resolved[i] < 0) resolved[i] = if (i == resolved.lastIndex) tokenCount.toLong() else 1L
        }
        require(resolved.fold(1L) { total, dimension -> total * dimension } == tokenCount.toLong()) {
            "Decoder input_ids must contain exactly $tokenCount codebook tokens"
        }
        return resolved
    }

    private fun tensorInfo(session: OrtSession, name: String): TensorInfo =
        session.inputInfo[name]?.info as? TensorInfo ?: throw IllegalArgumentException("$name is not a tensor input")

    private fun longTensor(environment: OrtEnvironment, values: LongArray, shape: LongArray): OnnxTensor =
        OnnxTensor.createTensor(environment, directLongBuffer(values), shape)

    private fun floatTensor(environment: OrtEnvironment, values: FloatArray, shape: LongArray): OnnxTensor =
        OnnxTensor.createTensor(environment, directFloatBuffer(values), shape)

    private fun directLongBuffer(values: LongArray): LongBuffer =
        ByteBuffer.allocateDirect(values.size * 8).order(ByteOrder.nativeOrder())
            .asLongBuffer().apply { put(values); flip() }

    private fun directFloatBuffer(values: FloatArray): FloatBuffer =
        ByteBuffer.allocateDirect(values.size * 4).order(ByteOrder.nativeOrder())
            .asFloatBuffer().apply { put(values); flip() }

    private fun OrtSession.Result.floatOutput(name: String): FloatOutput {
        val tensor = get(name).orElseThrow { IllegalStateException("ONNX output not found: $name") } as? OnnxTensor
            ?: throw IllegalStateException("ONNX output $name is not a tensor")
        val info = tensor.info as? TensorInfo ?: throw IllegalStateException("ONNX output $name is not a tensor")
        val buffer = tensor.floatBuffer
        val copy = FloatArray(buffer.remaining())
        buffer.get(copy)
        return FloatOutput(copy, info.shape)
    }

    private fun OrtSession.Result.firstTensorName(preferred: Set<String>): String {
        val names = map { it.key }.toList()
        return preferred.firstOrNull { it in names }
            ?: names.firstOrNull { get(it).orElse(null) is OnnxTensor }
            ?: throw IllegalStateException("ONNX graph returned no tensor outputs")
    }

    private fun checkCancelled(isCancelled: () -> Boolean) {
        if (isCancelled()) throw java.util.concurrent.CancellationException("Music generation cancelled")
    }

    private fun cacheSuffix(name: String): String = name
        .removePrefix("past_key_values")
        .removePrefix("present_key_values")
        .removePrefix("present")
        .trimStart('.', '_')
}
