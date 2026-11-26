package com.iris.localai.video

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaCodecList
import android.media.MediaFormat
import android.media.MediaMuxer
import java.io.File
import java.io.IOException
import java.nio.ByteBuffer

/** H.264 MP4 writer using flexible YUV input buffers and Android's native muxer. */
object Mp4VideoWriter {
    fun write(
        output: File,
        frames: List<Bitmap>,
        frameRate: Int,
        onProgress: (Int) -> Unit = {},
    ) {
        require(frames.isNotEmpty()) { "No frames to encode" }
        require(frameRate in 1..60) { "Unsupported video frame rate" }
        val width = frames.first().width
        val height = frames.first().height
        require(width % 2 == 0 && height % 2 == 0) { "H.264 output dimensions must be even" }
        require(frames.all { it.width == width && it.height == height }) { "Video frames must share dimensions" }
        if (output.exists()) throw IOException("Output file already exists: ${output.name}")
        val parent = output.absoluteFile.parentFile ?: throw IOException("Output directory is missing")
        if (!parent.isDirectory && !parent.mkdirs()) throw IOException("Unable to create video output directory")
        val temporary = File.createTempFile(".${output.nameWithoutExtension.take(40)}-", ".mp4", parent)
        var codec: MediaCodec? = null
        var muxer: MediaMuxer? = null
        var muxerStarted = false
        var complete = false
        try {
            val encoder = createEncoder(width, height, frameRate)
            codec = encoder
            val videoMuxer = MediaMuxer(temporary.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
            muxer = videoMuxer
            encoder.start()
            val info = MediaCodec.BufferInfo()
            var trackIndex = -1
            var outputDone = false

            fun drain(endOfStream: Boolean) {
                while (true) {
                    when (val index = encoder.dequeueOutputBuffer(info, if (endOfStream) 10_000 else 0)) {
                        MediaCodec.INFO_TRY_AGAIN_LATER -> return
                        MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                            if (muxerStarted) throw IOException("Encoder changed output format more than once")
                            trackIndex = videoMuxer.addTrack(encoder.outputFormat)
                            videoMuxer.start()
                            muxerStarted = true
                        }
                        MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED -> Unit
                        else -> if (index >= 0) {
                            val buffer = encoder.getOutputBuffer(index)
                                ?: throw IOException("Encoder returned an empty output buffer")
                            if (info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG != 0) info.size = 0
                            if (info.size > 0) {
                                if (!muxerStarted || trackIndex < 0) throw IOException("Encoder produced data before its format")
                                buffer.position(info.offset)
                                buffer.limit(info.offset + info.size)
                                videoMuxer.writeSampleData(trackIndex, buffer, info)
                            }
                            outputDone = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                            encoder.releaseOutputBuffer(index, false)
                            if (outputDone) return
                        }
                    }
                }
            }

            frames.forEachIndexed { index, bitmap ->
                val inputIndex = waitForInput(encoder)
                val image = encoder.getInputImage(inputIndex)
                    ?: throw IOException("H.264 encoder does not expose a flexible YUV input image")
                try {
                    writeYuv420(image.planes, bitmap)
                } finally {
                    image.close()
                }
                val presentationTimeUs = index * 1_000_000L / frameRate
                encoder.queueInputBuffer(inputIndex, 0, width * height * 3 / 2, presentationTimeUs, 0)
                drain(endOfStream = false)
                onProgress((index + 1) * 100 / frames.size)
            }

            val eosIndex = waitForInput(encoder)
            encoder.queueInputBuffer(eosIndex, 0, 0, frames.size * 1_000_000L / frameRate, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
            while (!outputDone) drain(endOfStream = true)
            if (!muxerStarted) throw IOException("H.264 encoder did not produce an MP4 track")
            videoMuxer.stop()
            muxerStarted = false
            if (output.exists() || !temporary.renameTo(output)) throw IOException("Unable to finalize MP4 file")
            complete = true
        } finally {
            if (muxerStarted) runCatching { muxer?.stop() }
            runCatching { codec?.stop() }
            runCatching { muxer?.release() }
            runCatching { codec?.release() }
            if (!complete) temporary.delete()
        }
    }

    private fun createEncoder(width: Int, height: Int, frameRate: Int): MediaCodec {
        val codecInfo = MediaCodecList(MediaCodecList.REGULAR_CODECS).codecInfos.firstOrNull { info ->
            info.isEncoder && info.supportedTypes.any { it.equals(MediaFormat.MIMETYPE_VIDEO_AVC, ignoreCase = true) } &&
                runCatching {
                    MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible in
                        info.getCapabilitiesForType(MediaFormat.MIMETYPE_VIDEO_AVC).colorFormats
                }.getOrDefault(false)
        } ?: throw IOException("No H.264 encoder with flexible YUV input is available on this device")
        val format = MediaFormat.createVideoFormat(MediaFormat.MIMETYPE_VIDEO_AVC, width, height).apply {
            setInteger(MediaFormat.KEY_COLOR_FORMAT, MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
            setInteger(MediaFormat.KEY_BIT_RATE, (width * height * frameRate / 5).coerceIn(500_000, 8_000_000))
            setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 2)
        }
        return MediaCodec.createByCodecName(codecInfo.name).apply { configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE) }
    }

    private fun waitForInput(codec: MediaCodec): Int {
        repeat(300) {
            val index = codec.dequeueInputBuffer(10_000)
            if (index >= 0) return index
        }
        throw IOException("Timed out waiting for H.264 encoder input")
    }

    private fun writeYuv420(planes: Array<android.media.Image.Plane>, bitmap: Bitmap) {
        require(planes.size == 3) { "Encoder did not provide three YUV planes" }
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)
        val y = planes[0]
        val u = planes[1]
        val v = planes[2]
        val yBuffer = y.buffer
        val uBuffer = u.buffer
        val vBuffer = v.buffer
        for (row in 0 until height) {
            for (column in 0 until width) {
                val pixel = pixels[row * width + column]
                val r = (pixel shr 16) and 255
                val g = (pixel shr 8) and 255
                val b = pixel and 255
                val value = (((66 * r + 129 * g + 25 * b + 128) shr 8) + 16).coerceIn(0, 255)
                yBuffer.put(row * y.rowStride + column * y.pixelStride, value.toByte())
                if (row % 2 == 0 && column % 2 == 0) {
                    val chromaU = (((-38 * r - 74 * g + 112 * b + 128) shr 8) + 128).coerceIn(0, 255)
                    val chromaV = (((112 * r - 94 * g - 18 * b + 128) shr 8) + 128).coerceIn(0, 255)
                    val chromaIndexU = (row / 2) * u.rowStride + (column / 2) * u.pixelStride
                    val chromaIndexV = (row / 2) * v.rowStride + (column / 2) * v.pixelStride
                    uBuffer.put(chromaIndexU, chromaU.toByte())
                    vBuffer.put(chromaIndexV, chromaV.toByte())
                }
            }
        }
    }
}
