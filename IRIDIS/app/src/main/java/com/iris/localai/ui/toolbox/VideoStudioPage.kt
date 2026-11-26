package com.iris.localai.ui.toolbox

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.content.ContentResolver
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.widget.MediaController
import android.widget.VideoView
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.ui.draw.clip
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.Alignment
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import com.iris.localai.onnx.OnnxBundle
import com.iris.localai.onnx.OnnxBundleLibrary
import com.iris.localai.ui.components.IrisButton
import com.iris.localai.ui.components.IrisButtonStyle
import com.iris.localai.ui.components.Fold
import com.iris.localai.ui.components.SectionTitle
import com.iris.localai.ui.components.SettingsGroup
import com.iris.localai.ui.components.SliderRow
import com.iris.localai.ui.icons.IrisIcons
import com.iris.localai.ui.theme.Iris
import com.iris.localai.ui.theme.IrisDimens
import com.iris.localai.ui.theme.IrisRadius
import com.iris.localai.ui.theme.IrisType
import com.iris.localai.video.VideoOnnxEngine
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.util.UUID
import android.widget.ImageView

@Composable
fun VideoStudioPage() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var bundles by remember { mutableStateOf<List<OnnxBundle>>(emptyList()) }
    var selectedId by rememberSaveable { mutableStateOf<String?>(null) }
    var requiredFilesExpanded by rememberSaveable { mutableStateOf(false) }
    var imageUri by rememberSaveable { mutableStateOf<Uri?>(null) }
    var output by remember { mutableStateOf<File?>(null) }
    var motionLevel by rememberSaveable { mutableFloatStateOf(1f) }
    var busy by remember { mutableStateOf(false) }
    var progress by remember { mutableIntStateOf(0) }
    var error by remember { mutableStateOf<String?>(null) }
    var statusMessage by remember { mutableStateOf<String?>(null) }
    val clipboard = LocalClipboardManager.current
    val mainHandler = remember { Handler(Looper.getMainLooper()) }

    suspend fun reload() {
        val loaded = withContext(Dispatchers.IO) { OnnxBundleLibrary.list(context.filesDir) }
        bundles = loaded
        val supported = loaded.filter { it.supportsVideoGeneration() }
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
            statusMessage = null
            try {
                val bundle = withContext(Dispatchers.IO) { OnnxBundleLibrary.importTree(context, uri) }
                reload()
                if (bundle.supportsVideoGeneration()) selectedId = bundle.id
                else error = "這個資料夾還不能用來生成影片，請查看上方所需檔案清單。"
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
        output = null
        error = null
    }

    val videoBundles = bundles.filter { it.supportsVideoGeneration() }
    val selected = videoBundles.firstOrNull { it.id == selectedId }
    val motionBucketId = when (motionLevel.toInt()) {
        0 -> 64
        2 -> 200
        else -> 127
    }

    SectionTitle("影片生成工作室")
    SettingsGroup {
        Column(
            Modifier.fillMaxWidth().padding(IrisDimens.space16),
            verticalArrangement = Arrangement.spacedBy(IrisDimens.space10),
        ) {
            VideoStepHeading("1", "載入影片模型")
            VideoModelStatusBadge(selected != null)
            selected?.let { Text("目前使用：${it.name}", style = IrisType.secondary, color = Iris.colors.textSecondary) }
            IrisButton(
                "選擇模型資料夾",
                onClick = { folderPicker.launch(null) },
                enabled = !busy,
                modifier = Modifier.fillMaxWidth(),
                icon = IrisIcons.Folder,
            )
            if (videoBundles.isNotEmpty()) {
                Text("已匯入的模型", style = IrisType.caption, color = Iris.colors.textSecondary)
                videoBundles.forEach { bundle ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(IrisDimens.space8),
                    ) {
                        IrisButton(
                            bundle.name,
                            onClick = { selectedId = bundle.id },
                            modifier = Modifier.weight(1f),
                            compact = true,
                            style = if (selectedId == bundle.id) IrisButtonStyle.Tonal else IrisButtonStyle.Outline,
                            icon = if (selectedId == bundle.id) IrisIcons.Check else null,
                        )
                        IrisButton(
                            "移除",
                            onClick = {
                                scope.launch {
                                    try {
                                        withContext(Dispatchers.IO) { OnnxBundleLibrary.delete(context.filesDir, bundle.id) }
                                        reload()
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
            IrisButton(
                if (requiredFilesExpanded) "隱藏所需檔案清單" else "查看所需檔案清單（點此展開）",
                onClick = { requiredFilesExpanded = !requiredFilesExpanded },
                compact = true,
                style = IrisButtonStyle.Text,
                icon = IrisIcons.ChevronRight,
            )
            Fold(requiredFilesExpanded) {
                Column(verticalArrangement = Arrangement.spacedBy(IrisDimens.space6)) {
                    VIDEO_REQUIRED_FILES.forEach { Text("• $it", style = IrisType.caption, color = Iris.colors.textSecondary) }
                }
            }
            IrisButton(
                "複製推薦模型名稱（SVD-XT ONNX）",
                onClick = {
                    clipboard.setText(AnnotatedString(RECOMMENDED_VIDEO_MODEL))
                    statusMessage = "已複製模型名稱，可貼上搜尋"
                },
                compact = true,
                style = IrisButtonStyle.Tonal,
                icon = IrisIcons.Copy,
            )
        }
    }

    SettingsGroup {
        Column(
            Modifier.fillMaxWidth().padding(IrisDimens.space16),
            verticalArrangement = Arrangement.spacedBy(IrisDimens.space10),
        ) {
            VideoStepHeading("2", "選擇首幀與動態")
            Text("先選一張圖片作為影片開場畫面。", style = IrisType.caption, color = Iris.colors.textSecondary)
            if (imageUri == null) {
                Box(
                    Modifier.fillMaxWidth().aspectRatio(16f / 9f)
                        .background(Iris.colors.surfaceVariant, IrisRadius.mediumShape),
                    contentAlignment = Alignment.Center,
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(IrisDimens.space8),
                    ) {
                        Icon(IrisIcons.Image, null, tint = Iris.colors.textTertiary, modifier = Modifier.size(32.dp))
                        Text("尚未選取首幀圖片", style = IrisType.caption, color = Iris.colors.textTertiary)
                    }
                }
            } else {
                AndroidView(
                    modifier = Modifier.fillMaxWidth().aspectRatio(16f / 9f)
                        .clip(IrisRadius.mediumShape)
                        .background(Iris.colors.surfaceVariant, IrisRadius.mediumShape),
                    factory = { viewContext ->
                        ImageView(viewContext).apply {
                            scaleType = ImageView.ScaleType.CENTER_CROP
                            setImageURI(imageUri)
                            contentDescription = "影片首幀預覽"
                        }
                    },
                    update = { it.setImageURI(imageUri) },
                )
                Text(
                    imageUri?.lastPathSegment?.substringAfterLast('/') ?: "已選取圖片",
                    style = IrisType.caption,
                    color = Iris.colors.textTertiary,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
            IrisButton(
                if (imageUri == null) "從相簿選取首幀" else "更換首幀圖片",
                onClick = { imagePicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                enabled = !busy,
                modifier = Modifier.fillMaxWidth(),
                style = IrisButtonStyle.Tonal,
                icon = IrisIcons.Image,
            )
            SliderRow(
                "動態強度",
                motionLevel,
                0f..2f,
                when (motionBucketId) {
                    64 -> "微動"
                    200 -> "大幅度"
                    else -> "標準"
                },
                subtitle = "調整畫面變化幅度",
                steps = 1,
            ) { motionLevel = it }
        }
    }

    SettingsGroup {
        Column(
            Modifier.fillMaxWidth().padding(IrisDimens.space16),
            verticalArrangement = Arrangement.spacedBy(IrisDimens.space10),
        ) {
            VideoStepHeading("3", "生成與觀看影片")
            Text(
                when {
                    busy -> "正在生成影片… $progress%"
                    output != null -> "影片已完成"
                    else -> "生成後可直接在這裡播放"
                },
                style = IrisType.secondary,
                color = if (output != null) Iris.colors.success else Iris.colors.textSecondary,
            )
            LinearProgressIndicator(
                progress = { if (output != null) 1f else (progress / 100f).coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
            )
            IrisButton(
                if (busy) "正在生成… $progress%" else "開始生成 MP4 影片",
                onClick = {
                    val uri = imageUri
                    val bundle = selected
                    if (uri != null && bundle != null) scope.launch {
                        busy = true
                        progress = 0
                        error = null
                        statusMessage = null
                        output = null
                        try {
                            val generated = withContext(Dispatchers.IO) {
                                val bitmap = decodeBitmap(context.contentResolver, uri)
                                try {
                                    val file = File(context.filesDir, "videos/${UUID.randomUUID()}.mp4")
                                    VideoOnnxEngine.generateMp4(
                                        context.filesDir,
                                        bundle,
                                        bitmap,
                                        file,
                                        motionBucketId = motionBucketId,
                                        onProgress = { value -> mainHandler.post { progress = value } },
                                    )
                                } finally {
                                    bitmap.recycle()
                                }
                            }
                            output = generated
                            progress = 100
                        } catch (e: CancellationException) {
                            throw e
                        } catch (e: Exception) {
                            error = e.localizedMessage ?: e.javaClass.simpleName
                        } finally {
                            busy = false
                        }
                    }
                },
                enabled = !busy && selected != null && imageUri != null,
                modifier = Modifier.fillMaxWidth(),
                icon = IrisIcons.Movie,
            )
            if (selected == null) Text("請先完成第 1 步：選擇影片模型資料夾。", style = IrisType.caption, color = Iris.colors.textTertiary)
            if (imageUri == null) Text("請先完成第 2 步：選取一張首幀圖片。", style = IrisType.caption, color = Iris.colors.textTertiary)
            output?.let { file ->
                Text(file.name, style = IrisType.caption, color = Iris.colors.textTertiary, maxLines = 1, overflow = TextOverflow.Ellipsis)
                VideoPreview(file)
            }
        }
    }
    statusMessage?.let { Text(it, style = IrisType.caption, color = Iris.colors.success) }
    error?.let { Text("發生問題：$it", style = IrisType.caption, color = Iris.colors.warning) }
}

private val VIDEO_REQUIRED_FILES = listOf(
    "vae_encoder.onnx",
    "image_encoder.onnx",
    "unet.onnx",
    "vae_decoder.onnx",
    "scheduler_config.json",
)
private const val RECOMMENDED_VIDEO_MODEL = "SVD-XT ONNX"

private fun OnnxBundle.supportsVideoGeneration(): Boolean {
    val names = assets.map { it.path.substringAfterLast('/') }.toSet()
    return names.containsAll(VIDEO_REQUIRED_FILES)
}

@Composable
private fun VideoStepHeading(number: String, title: String) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(IrisDimens.space10)) {
        Box(
            Modifier.size(32.dp).background(Iris.colors.accentContainer, IrisRadius.mediumShape),
            contentAlignment = Alignment.Center,
        ) {
            Text(number, style = IrisType.label, color = Iris.colors.onAccentContainer)
        }
        Text(title, style = IrisType.label, color = Iris.colors.text)
    }
}

@Composable
private fun VideoModelStatusBadge(ready: Boolean) {
    val color = if (ready) Iris.colors.success else Iris.colors.warning
    Text(
        if (ready) "✅ 模型已就緒" else "⚠️ 尚未載入影片模型",
        style = IrisType.label,
        color = color,
        modifier = Modifier.fillMaxWidth()
            .background(color.copy(alpha = 0.12f), IrisRadius.mediumShape)
            .padding(horizontal = IrisDimens.space12, vertical = IrisDimens.space10),
    )
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
