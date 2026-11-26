package com.iris.localai.musicgen

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.File
import java.io.IOException
import java.nio.file.Files

class MusicGenFoundationTest {
    private fun tempDir(): File = Files.createTempDirectory("musicgen-test").toFile()

    @Test
    fun importsArbitraryLocalFilesAndPersistsManifest() {
        val root = tempDir()
        val imported = MusicModelLibrary.import(
            root,
            listOf(
                MusicModelSource("my decoder.gguf") { ByteArrayInputStream(byteArrayOf(1, 2, 3)) },
                MusicModelSource("audio codec.safetensors") { ByteArrayInputStream(byteArrayOf(4, 5)) },
                MusicModelSource("weights.mnn") { ByteArrayInputStream(byteArrayOf(6)) },
            ),
            "  My   Music Model  ",
        )

        assertEquals("My Music Model", imported.name)
        assertEquals(listOf("gguf", "safetensors", "mnn"), imported.assets.map { it.format })
        assertEquals(listOf(3L, 2L, 1L), imported.assets.map { it.sizeBytes })
        assertEquals(listOf(imported), MusicModelLibrary.list(root))
        assertTrue(File(File(root, MusicModelLibrary.MODELS_SUBDIR), imported.id).isDirectory)
    }

    @Test
    fun rejectsUnsupportedAndDuplicateNamesBeforeImport() {
        val root = tempDir()
        assertThrows(IllegalArgumentException::class.java) {
            MusicModelLibrary.import(root, listOf(MusicModelSource("weights.pt") { ByteArrayInputStream(byteArrayOf(1)) }))
        }
        assertThrows(IllegalArgumentException::class.java) {
            MusicModelLibrary.import(
                root,
                listOf(
                    MusicModelSource("model.gguf") { ByteArrayInputStream(byteArrayOf(1)) },
                    MusicModelSource("MODEL.GGUF") { ByteArrayInputStream(byteArrayOf(2)) },
                ),
            )
        }
        assertTrue(MusicModelLibrary.list(root).isEmpty())
    }

    @Test
    fun failedImportCleansItsIncompleteModelDirectory() {
        val root = tempDir()
        assertThrows(IOException::class.java) {
            MusicModelLibrary.import(
                root,
                listOf(
                    MusicModelSource("first.gguf") { ByteArrayInputStream(byteArrayOf(1)) },
                    MusicModelSource("broken.mnn") { throw IOException("source read failed") },
                ),
            )
        }
        assertTrue(MusicModelLibrary.list(root).isEmpty())
    }

    @Test
    fun modelDeleteValidatesId() {
        val root = tempDir()
        val model = MusicModelLibrary.import(root, listOf(MusicModelSource("model.gguf") { ByteArrayInputStream(byteArrayOf(1)) }))
        MusicModelLibrary.delete(root, model.id)
        assertTrue(MusicModelLibrary.list(root).isEmpty())
        assertThrows(IllegalArgumentException::class.java) { MusicModelLibrary.delete(root, "../other") }
    }

    @Test
    fun writesMonoPcm16WavAt32Khz() {
        val output = File(tempDir(), "music.wav")
        PcmWavWriter.writeMono16(output, floatArrayOf(-1f, -2f, 0f, 1f, 2f, Float.NaN))
        val wav = output.readBytes()
        assertEquals(56, wav.size)
        assertArrayEquals("RIFF".toByteArray(), wav.copyOfRange(0, 4))
        assertEquals(48, le32(wav, 4))
        assertArrayEquals("WAVEfmt ".toByteArray(), wav.copyOfRange(8, 16))
        assertEquals(32_000, le32(wav, 24))
        assertEquals(64_000, le32(wav, 28))
        assertEquals(16, le16(wav, 34))
        assertArrayEquals("data".toByteArray(), wav.copyOfRange(36, 40))
        assertEquals(12, le32(wav, 40))
        assertArrayEquals(
            byteArrayOf(0, 128, 0, 128, 0, 0, -1, 127, -1, 127, 0, 0),
            wav.copyOfRange(44, 56),
        )
    }

    @Test
    fun wavRejectsInvalidRateAndExistingFile() {
        val output = File(tempDir(), "music.wav")
        assertThrows(IllegalArgumentException::class.java) { PcmWavWriter.writeMono16(output, floatArrayOf(), 1_000) }
        PcmWavWriter.writeMono16(output, floatArrayOf())
        assertThrows(IllegalArgumentException::class.java) { PcmWavWriter.writeMono16(output, floatArrayOf()) }
        assertEquals(listOf("music.wav"), output.parentFile.list()?.sorted())
    }

    private fun le16(bytes: ByteArray, offset: Int): Int =
        (bytes[offset].toInt() and 0xff) or ((bytes[offset + 1].toInt() and 0xff) shl 8)

    private fun le32(bytes: ByteArray, offset: Int): Int =
        le16(bytes, offset) or (le16(bytes, offset + 2) shl 16)
}
