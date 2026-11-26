package com.iris.localai.ui.toolbox

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.content.ContentResolver
import android.media.MediaController
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.widget.VideoView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.iris.localai.onnx.OnnxBundle
import com.iris.localai.onnx.OnnxBundleLibrary
import com.iris.localai.ui.components.IrisButton
import com.iris.localai.ui.components.IrisButtonStyle
import com.iris.localai.ui.components.SectionTitle
import com.iris.localai.ui.components.SettingsGroup
import com.iris.localai.ui.theme.Iris
import com.iris.localai.ui.theme.IrisDimens
import com.iris.localai.ui.theme.IrisType
import com.iris.localai.video.VideoOnnxEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.util.UUID

@Composable
fun VideoStudioPage() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var bundles by remember { mutableStateOf<List<OnnxBundle>>(emptyList()) }
    var selectedId by rememberSaveable { mutableStateOf<String?>(null) }
    var imageUri by rememberSaveable { mutableStateOf<Uri?>(null) }
    var output by remember { mutableStateOf<File?>(null) }
    var busy by remember { mutableStateOf(false) }
    var progress by remember { mutableIntStateOf(0) }
    var error by remember { mutableStateOf<String?>(null) }

    suspend fun reload() {
        val loaded = withContext(Dispatchers.IO) { OnnxBundleLibrary.list(context.filesDir) }
        bundles = loaded
        val supported = loaded.filter { bundle ->
            val names = bundle.assets.map { it.path.substringAfterLast('/') }.toSet()
            setOf("vae_encoder.onnx", "image_encoder.onnx", "unet.onnx", "vae_decoder.onnx", "scheduler_config.json")
                .all { it in names }
        }
        if (selectedId !in supported.map { it.id }) selectedId = supported.firstOrNull()?.id
    }

    LaunchedEffect(context.filesDir) {
        try {
            reload()
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            error = e.localizedMessage ?: e.javaClass.simpleName
        }
    }

    val folderPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) scope.launch {
            busy = true
            error = null
            try {
                val bundle = withContext(Dispatchers.IO) { OnnxBundleLibrary.importTree(context, uri) }
                reload()
                selectedId = bundle.id
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                error = e.localizedMessage ?: e.javaClass.simpleName
            } finally {
                busy = false
            }
        }
    }
    val imagePicker = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        imageUri = uri
        error = null
    }

    val videoBundles = bundles.filter { bundle ->
        val names = bundle.assets.map { it.path.substringAfterLast('/') }.toSet()
        setOf("vae_encoder.onnx", "image_encoder.onnx", "unet.onnx", "vae_decoder.onnx", "scheduler_config.json")
            .all { it in names }
    }
    val selected = videoBundles.firstOrNull { it.id == selectedId }
    SectionTitle("SVD-XT 圖生影片")
    SettingsGroup {
        Column(Modifier.fillMaxWidth().padding(IrisDimens.space16)) {
            Text(
                "模型資料夾需包含 vae_encoder.onnx、image_encoder.onnx、unet.onnx、vae_decoder.onnx 與 scheduler_config.json。",
                style = IrisType.secondary,
                color = Iris.colors.textSecondary,
            )
            Spacer(Modifier.height(IrisDimens.space8))
            Text(
                "使用首幀圖片離線推論，輸出 512×288、14 幀、6 FPS 的 H.264 MP4。模型權重保留在本機。",
                style = IrisType.caption,
                color = Iris.colors.textTertiary,
            )
            Spacer(Modifier.height(IrisDimens.space12))
            IrisButton(
                "匯入 SVD-XT 模型資料夾",
                onClick = { folderPicker.launch(null) },
                enabled = !busy,
                compact = true,
                style = IrisButtonStyle.Tonal,
            )
        }
    }
    videoBundles.forEach { bundle ->
        SettingsGroup {
            Column(
                Modifier.fillMaxWidth()
                    .clickable { selectedId = bundle.id }
                    .padding(IrisDimens.space16),
            ) {
                Text(bundle.name, style = IrisType.label, color = Iris.colors.text)
                Text("${bundle.assets.size} 個模型檔案", style = IrisType.caption, color = Iris.colors.textSecondary)
                if (selectedId == bundle.id) Text("已選取", style = IrisType.caption, color = Iris.colors.accent)
                IrisButton(
                    "刪除模型",
                    onClick = {
                        scope.launch {
                            try {
                                withContext(Dispatchers.IO) { OnnxBundleLibrary.delete(context.filesDir, bundle.id) }
                                reload()
                                error = null
                            } catch (e: CancellationException) {
                                throw e
                            } catch (e: Exception) {
                                error = e.localizedMessage ?: e.javaClass.simpleName
                            }
                        }
                    },
                    compact = true,
                    style = IrisButtonStyle.Text,
                    destructive = true,
                )
            }
        }
    }

    SectionTitle("首幀圖片")
    SettingsGroup {
        Column(Modifier.fillMaxWidth().padding(IrisDimens.space16)) {
            Text(imageUri?.lastPathSegment ?: "尚未選取圖片", style = IrisType.secondary, color = Iris.colors.textSecondary)
            IrisButton(
                "從相簿選取圖片",
                onClick = { imagePicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                enabled = !busy,
                compact = true,
                style = IrisButtonStyle.Tonal,
            )
            IrisButton(
                if (busy) "正在生成… $progress%" else "生成 MP4",
                onClick = {
                    val uri = imageUri
                    val bundle = selected
                    if (uri != null && bundle != null) {
                        scope.launch {
                            busy = true
                            progress = 0
                            error = null
                            output = null
                            var bitmap: Bitmap? = null
                            try {
                                output = withContext(Dispatchers.IO) {
                                    bitmap = decodeBitmap(context.contentResolver, uri)
                                    val file = File(context.filesDir, "videos/${UUID.randomUUID()}.mp4")
                                    VideoOnnxEngine.generateMp4(
                                        context.filesDir,
                                        bundle,
                                        bitmap!!,
                                        file,
                                        onProgress = { value -> Handler(Looper.getMainLooper()).post { progress = value } },
                                    )
                                }
                            } catch (e: CancellationException) {
                                throw e
                            } catch (e: Exception) {
                                error = e.localizedMessage ?: e.javaClass.simpleName
                            } finally {
                                bitmap?.recycle()
                                busy = false
                            }
                        }
                    }
                },
                enabled = !busy && selected != null && imageUri != null,
                compact = true,
                style = IrisButtonStyle.Primary,
            )
        }
    }
    output?.let { file ->
        SectionTitle("生成影片")
        SettingsGroup {
            Column(Modifier.fillMaxWidth().padding(IrisDimens.space16)) {
                Text(file.name, style = IrisType.label, color = Iris.colors.text)
                VideoPreview(file)
            }
        }
    }
    error?.let { Text("錯誤：$it", style = IrisType.caption, color = Iris.colors.warning) }
}

@Composable
private fun VideoPreview(file: File) {
    val context = LocalContext.current
    val controller = remember(file) { MediaController(context) }
    var videoView by remember(file) { mutableStateOf<VideoView?>(null) }
    DisposableEffect(file) {
        onDispose {
            videoView?.stopPlayback()
            controller.hide()
        }
    }
    AndroidView(
        modifier = Modifier.fillMaxWidth().height(240.dp),
        factory = { viewContext ->
            VideoView(viewContext).apply {
                videoView = this
                setMediaController(controller)
                controller.setAnchorView(this)
                setVideoURI(Uri.fromFile(file))
            }
        },
        update = { view ->
            if (view.tag != file.absolutePath) {
                view.tag = file.absolutePath
                view.setVideoURI(Uri.fromFile(file))
            }
        },
    )
}

private fun decodeBitmap(resolver: ContentResolver, uri: Uri): Bitmap {
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    resolver.openInputStream(uri)?.use { stream ->
        BitmapFactory.decodeStream(stream, null, bounds)
    } ?: throw IOException("無法讀取所選圖片")
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw IOException("所選檔案不是有效圖片")
    val options = BitmapFactory.Options().apply {
        var sample = 1
        while (bounds.outWidth / sample > 1536 || bounds.outHeight / sample > 1536) sample *= 2
        inSampleSize = sample
    }
    return resolver.openInputStream(uri)?.use { stream ->
        BitmapFactory.decodeStream(stream, null, options)
    } ?: throw IOException("無法解碼所選圖片")
}
