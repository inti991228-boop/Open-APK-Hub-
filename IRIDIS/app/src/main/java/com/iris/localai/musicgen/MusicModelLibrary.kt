package com.iris.localai.musicgen

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.util.UUID

@Serializable
data class MusicModelAsset(
    val fileName: String,
    val format: String,
    val sizeBytes: Long,
)

@Serializable
data class ImportedMusicModel(
    val id: String,
    val name: String,
    val importedAt: Long,
    val assets: List<MusicModelAsset>,
)

data class MusicModelSource(
    val fileName: String,
    val open: () -> InputStream,
)

object MusicModelLibrary {
    const val MODELS_SUBDIR = "models/musicgen"
    private val idPattern = Regex("mg-[0-9a-f]{32}")
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }
    private val supportedFormats = setOf("gguf", "safetensors", "mnn")

    fun formatOf(fileName: String): String? =
        fileName.substringAfterLast('.', "").lowercase().takeIf { it in supportedFormats }

    fun list(filesDir: File): List<ImportedMusicModel> {
        val root = File(filesDir, MODELS_SUBDIR)
        if (!root.exists()) return emptyList()
        val dirs = root.listFiles() ?: throw IOException("Unable to read MusicGen model directory")
        return dirs.asSequence()
            .filter { it.isDirectory && idPattern.matches(it.name) }
            .map { dir ->
                val manifest = File(dir, "manifest.json")
                if (!manifest.isFile) throw IOException("Missing model manifest: ${dir.name}")
                val model = json.decodeFromString<ImportedMusicModel>(manifest.readText())
                if (model.id != dir.name || model.assets.isEmpty() || model.assets.size > 64) {
                    throw IOException("Invalid model manifest: ${dir.name}")
                }
                model.assets.forEach { asset ->
                    val assetFile = File(dir, asset.fileName)
                    if (safeFileName(asset.fileName) != asset.fileName ||
                        formatOf(asset.fileName) != asset.format ||
                        !assetFile.isFile ||
                        assetFile.length() != asset.sizeBytes
                    ) {
                        throw IOException("Missing or incomplete model file: ${asset.fileName}")
                    }
                }
                model
            }
            .sortedByDescending { it.importedAt }
            .toList()
    }

    fun import(filesDir: File, sources: List<MusicModelSource>, displayName: String? = null): ImportedMusicModel {
        require(sources.isNotEmpty()) { "Select at least one model file" }
        require(sources.size <= 64) { "Select no more than 64 model files at once" }
        val normalized = sources.map { source ->
            val name = safeFileName(source.fileName)
            val format = formatOf(name) ?: throw IllegalArgumentException("Unsupported model file: $name")
            source to MusicModelAsset(name, format, 0)
        }
        require(normalized.map { it.second.fileName.lowercase() }.distinct().size == normalized.size) {
            "Selected files must have unique names"
        }

        val root = File(filesDir, MODELS_SUBDIR)
        if (!root.isDirectory && !root.mkdirs()) throw IOException("Unable to create MusicGen model directory")
        val id = "mg-${UUID.randomUUID().toString().replace("-", "")}"
        val dir = File(root, id)
        if (!dir.mkdir()) throw IOException("Unable to create imported model directory")
        var complete = false
        try {
            val assets = normalized.map { (source, asset) ->
                val target = File(dir, asset.fileName)
                source.open().use { input ->
                    FileOutputStream(target).use { output -> input.copyTo(output) }
                }
                asset.copy(sizeBytes = target.length())
            }
            val name = cleanName(displayName ?: nameFromFile(assets.first().fileName))
            val model = ImportedMusicModel(id, name, System.currentTimeMillis(), assets)
            val manifest = File(dir, "manifest.json")
            val tempManifest = File(dir, "manifest.json.tmp")
            tempManifest.writeText(json.encodeToString(model))
            if (!tempManifest.renameTo(manifest)) throw IOException("Unable to save model manifest")
            complete = true
            return model
        } finally {
            if (!complete) dir.deleteRecursively()
        }
    }

    fun delete(filesDir: File, id: String) {
        require(idPattern.matches(id)) { "Invalid model ID" }
        val dir = File(File(filesDir, MODELS_SUBDIR), id)
        if (dir.exists() && !dir.deleteRecursively()) throw IOException("Unable to delete model: $id")
    }

    fun safeFileName(name: String): String =
        name.substringAfterLast('/').substringAfterLast('\\')
            .replace(Regex("[\\\\:*?\"<>|\\u0000-\\u001f]"), "_")
            .trim().trim('.').take(120)
            .ifEmpty { "model.bin" }

    fun nameFromFile(fileName: String): String =
        fileName.substringBeforeLast('.', fileName).replace('_', ' ').trim().take(40).ifEmpty { fileName }

    fun cleanName(name: String): String =
        name.trim().replace(Regex("\\s+"), " ").take(40).ifEmpty { "MusicGen model" }
}
