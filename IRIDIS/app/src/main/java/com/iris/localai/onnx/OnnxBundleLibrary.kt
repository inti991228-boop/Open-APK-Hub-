package com.iris.localai.onnx

import android.content.Context
import android.net.Uri
import android.os.StatFs
import android.provider.DocumentsContract
import kotlinx.serialization.Serializable
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.UUID

@Serializable
data class OnnxBundleAsset(val path: String, val sizeBytes: Long)

@Serializable
data class OnnxBundle(
    val id: String,
    val name: String,
    val importedAt: Long,
    val assets: List<OnnxBundleAsset>,
)

object OnnxBundleLibrary {
    const val ROOT = "models/onnx-bundles"
    private const val MAX_FILES = 64
    private const val MAX_DEPTH = 8
    private val idPattern = Regex("onnx-[0-9a-f]{32}")
    private val allowedExtensions = setOf("onnx", "onnx_data", "data", "bin", "json", "txt", "model", "vocab")
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    fun list(filesDir: File): List<OnnxBundle> {
        val root = File(filesDir, ROOT)
        if (!root.exists()) return emptyList()
        val dirs = root.listFiles() ?: throw IOException("Unable to read ONNX model directory")
        return dirs.asSequence().filter { it.isDirectory && idPattern.matches(it.name) }.map { dir ->
            val manifestFile = File(dir, "manifest.json")
            if (!manifestFile.isFile) throw IOException("Missing model manifest: ${dir.name}")
            val bundle = json.decodeFromString<OnnxBundle>(manifestFile.readText())
            if (bundle.id != dir.name || bundle.assets.isEmpty() || bundle.assets.size > MAX_FILES ||
                bundle.assets.map { it.path }.distinct().size != bundle.assets.size
            ) {
                throw IOException("Invalid ONNX model manifest: ${dir.name}")
            }
            bundle.assets.forEach { asset ->
                if (!safeRelativePath(asset.path)) throw IOException("Unsafe model asset path")
                val file = File(dir, asset.path)
                if (!file.isFile || file.length() != asset.sizeBytes) {
                    throw IOException("Missing or incomplete model file: ${asset.path}")
                }
            }
            bundle
        }.sortedByDescending { it.importedAt }.toList()
    }

    fun importTree(context: Context, treeUri: Uri, displayName: String? = null): OnnxBundle {
        val resolver = context.contentResolver
        val treeId = DocumentsContract.getTreeDocumentId(treeUri)
        val files = ArrayList<TreeFile>()
        collectFiles(context, treeUri, treeId, "", 0, files)
        require(files.isNotEmpty()) { "The selected folder contains no ONNX bundle files" }
        require(files.size <= MAX_FILES) { "A model bundle may contain at most $MAX_FILES supported files" }
        require(files.map { it.relativePath.lowercase() }.distinct().size == files.size) { "Duplicate model file paths" }
        var totalBytes = 0L
        files.forEach { entry ->
            val size = entry.sizeBytes.coerceAtLeast(0L)
            if (Long.MAX_VALUE - totalBytes < size) throw IOException("Model bundle size is too large")
            totalBytes += size
        }
        val available = StatFs(context.filesDir.absolutePath).availableBytes
        if (totalBytes > 0 && totalBytes > available) {
            throw IOException("Not enough app storage for this model bundle")
        }

        val root = File(context.filesDir, ROOT)
        if (!root.isDirectory && !root.mkdirs()) throw IOException("Unable to create ONNX model directory")
        val id = "onnx-${UUID.randomUUID().toString().replace("-", "")}"
        val dir = File(root, id)
        if (!dir.mkdir()) throw IOException("Unable to create imported model directory")
        var complete = false
        try {
            val copied = files.map { entry ->
                val target = File(dir, entry.relativePath)
                target.parentFile?.let { parent ->
                    if (!parent.isDirectory && !parent.mkdirs()) throw IOException("Unable to create model subdirectory")
                }
                resolver.openInputStream(entry.uri)?.use { input ->
                    FileOutputStream(target).use { output -> input.copyTo(output) }
                } ?: throw IOException("Unable to read model asset: ${entry.relativePath}")
                OnnxBundleAsset(entry.relativePath, target.length())
            }
            val name = cleanName(displayName ?: queryDisplayName(context, treeUri) ?: "ONNX model bundle")
            val bundle = OnnxBundle(id, name, System.currentTimeMillis(), copied)
            val temp = File(dir, "manifest.json.tmp")
            temp.writeText(json.encodeToString(bundle))
            if (!temp.renameTo(File(dir, "manifest.json"))) throw IOException("Unable to save ONNX model manifest")
            complete = true
            return bundle
        } finally {
            if (!complete) dir.deleteRecursively()
        }
    }

    fun delete(filesDir: File, id: String) {
        require(idPattern.matches(id)) { "Invalid ONNX model ID" }
        val dir = File(File(filesDir, ROOT), id)
        if (dir.exists() && !dir.deleteRecursively()) throw IOException("Unable to delete ONNX model: $id")
    }

    fun resolve(filesDir: File, bundle: OnnxBundle, fileName: String): File {
        val matches = bundle.assets.filter { it.path.substringAfterLast('/') == fileName }
        require(matches.size == 1) {
            if (matches.isEmpty()) "Model bundle is missing $fileName" else "Model bundle contains multiple files named $fileName"
        }
        val file = File(File(filesDir, ROOT), bundle.id).resolve(matches.single().path).canonicalFile
        val modelRoot = File(File(filesDir, ROOT), bundle.id).canonicalFile
        require(file.path.startsWith(modelRoot.path + File.separator)) { "Model asset is outside its bundle" }
        if (!file.isFile) throw IOException("Model file not found: $fileName")
        return file
    }

    private data class TreeFile(val uri: Uri, val relativePath: String, val sizeBytes: Long)

    private fun collectFiles(
        context: Context,
        treeUri: Uri,
        parentId: String,
        parentPath: String,
        depth: Int,
        files: MutableList<TreeFile>,
    ) {
        if (depth > MAX_DEPTH) throw IOException("Model folder nesting exceeds $MAX_DEPTH levels")
        val childrenUri = DocumentsContract.buildChildDocumentsUriUsingTree(treeUri, parentId)
        context.contentResolver.query(
            childrenUri,
            arrayOf(
                DocumentsContract.Document.COLUMN_DOCUMENT_ID,
                DocumentsContract.Document.COLUMN_DISPLAY_NAME,
                DocumentsContract.Document.COLUMN_MIME_TYPE,
                DocumentsContract.Document.COLUMN_SIZE,
            ),
            null,
            null,
            null,
        )?.use { cursor ->
            val idColumn = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DOCUMENT_ID)
            val nameColumn = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_DISPLAY_NAME)
            val mimeColumn = cursor.getColumnIndexOrThrow(DocumentsContract.Document.COLUMN_MIME_TYPE)
            val sizeColumn = cursor.getColumnIndex(DocumentsContract.Document.COLUMN_SIZE)
            while (cursor.moveToNext()) {
                val documentId = cursor.getString(idColumn)
                val name = cursor.getString(nameColumn)
                val mime = cursor.getString(mimeColumn)
                if (mime == DocumentsContract.Document.MIME_TYPE_DIR) {
                    collectFiles(context, treeUri, documentId, "$parentPath${safeSegment(name)}/", depth + 1, files)
                } else if (name != "manifest.json" &&
                    name.substringAfterLast('.', "").lowercase() in allowedExtensions
                ) {
                    val relative = "$parentPath${safeSegment(name)}"
                    if (!safeRelativePath(relative)) throw IOException("Unsafe model file name")
                    val size = if (sizeColumn >= 0 && !cursor.isNull(sizeColumn)) cursor.getLong(sizeColumn) else -1L
                    files += TreeFile(DocumentsContract.buildDocumentUriUsingTree(treeUri, documentId), relative, size)
                    if (files.size > MAX_FILES) throw IOException("A model bundle may contain at most $MAX_FILES supported files")
                }
            }
        } ?: throw IOException("Unable to list the selected model folder")
    }

    private fun queryDisplayName(context: Context, uri: Uri): String? =
        context.contentResolver.query(uri, arrayOf(DocumentsContract.Document.COLUMN_DISPLAY_NAME), null, null, null)
            ?.use { cursor -> if (cursor.moveToFirst()) cursor.getString(0) else null }

    private fun safeSegment(value: String): String =
        value.replace(Regex("[/\\\\:*?\"<>|\\u0000-\\u001f]"), "_").trim().trim('.')
            .take(120).ifEmpty { "model-file" }

    private fun safeRelativePath(path: String): Boolean =
        path.isNotBlank() && !path.startsWith('/') && path.split('/').all { it.isNotBlank() && it != "." && it != ".." }

    private fun cleanName(value: String): String =
        value.trim().replace(Regex("\\s+"), " ").take(60).ifEmpty { "ONNX model bundle" }
}
