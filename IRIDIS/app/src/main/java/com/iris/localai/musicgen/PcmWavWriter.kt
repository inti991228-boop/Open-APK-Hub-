package com.iris.localai.musicgen

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.roundToInt

object PcmWavWriter {
    fun writeMono16(output: File, samples: FloatArray, sampleRate: Int = 32_000): File {
        require(sampleRate in 8_000..192_000) { "Unsupported sample rate: $sampleRate" }
        require(samples.size.toLong() * 2 <= Int.MAX_VALUE - 36L) { "PCM data is too large for a WAV file" }
        require(!output.exists()) { "Output file already exists: ${output.name}" }
        val parent = output.absoluteFile.parentFile ?: throw IOException("Output directory is missing")
        if (!parent.isDirectory && !parent.mkdirs()) throw IOException("Unable to create output directory")
        val temp = File.createTempFile(".${output.name.take(40)}-", ".part", parent)
        var complete = false
        try {
            BufferedOutputStream(FileOutputStream(temp)).use { out ->
                val dataSize = samples.size * 2
                out.writeAscii("RIFF")
                out.writeLe32(36 + dataSize)
                out.writeAscii("WAVE")
                out.writeAscii("fmt ")
                out.writeLe32(16)
                out.writeLe16(1)
                out.writeLe16(1)
                out.writeLe32(sampleRate)
                out.writeLe32(sampleRate * 2)
                out.writeLe16(2)
                out.writeLe16(16)
                out.writeAscii("data")
                out.writeLe32(dataSize)
                samples.forEach { sample ->
                    val finite = if (sample.isFinite()) sample.coerceIn(-1f, 1f) else 0f
                    val pcm = if (finite <= -1f) Short.MIN_VALUE.toInt() else (finite * Short.MAX_VALUE).roundToInt()
                    out.write(pcm and 0xff)
                    out.write((pcm ushr 8) and 0xff)
                }
            }
            if (output.exists() || !temp.renameTo(output)) throw IOException("Unable to save WAV file")
            complete = true
            return output
        } finally {
            if (!complete) temp.delete()
        }
    }

    private fun java.io.OutputStream.writeAscii(value: String) = write(value.toByteArray(Charsets.US_ASCII))

    private fun java.io.OutputStream.writeLe16(value: Int) {
        write(value and 0xff)
        write((value ushr 8) and 0xff)
    }

    private fun java.io.OutputStream.writeLe32(value: Int) {
        write(value and 0xff)
        write((value ushr 8) and 0xff)
        write((value ushr 16) and 0xff)
        write((value ushr 24) and 0xff)
    }
}
