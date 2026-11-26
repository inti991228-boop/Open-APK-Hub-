package com.iris.localai.ui.toolbox
import androidx.compose.animation.AnimatedContent
import androidx.compose.foundation.text.selection.SelectionContainer
import com.iris.localai.ui.components.Fold
import com.iris.localai.ui.components.rememberLast
import com.iris.localai.ui.motion.IrisMotion
import com.iris.localai.ui.motion.motion
import com.iris.localai.ui.theme.LocalReduceMotion

import androidx.compose.ui.draw.clip

import androidx.compose.foundation.horizontalScroll

import android.content.ClipData
import android.content.Intent
import android.media.MediaPlayer
import android.net.Uri
import android.os.Handler
import android.os.Looper
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.material3.AssistChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.core.content.FileProvider
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.iris.localai.I18n
import com.iris.localai.R
import com.iris.localai.floating.FloatingService
import com.iris.localai.musicgen.MusicGenOnnxEngine
import com.iris.localai.onnx.OnnxBundle
import com.iris.localai.onnx.OnnxBundleLibrary
import com.iris.localai.ui.chat.ChatViewModel
import com.iris.localai.ui.components.IrisButton
import com.iris.localai.ui.components.IrisButtonStyle
import com.iris.localai.ui.components.IrisCard
import com.iris.localai.ui.components.ScrollBody
import com.iris.localai.ui.components.SectionTitle
import com.iris.localai.ui.components.SettingsGroup
import com.iris.localai.ui.components.SettingsRow
import com.iris.localai.ui.components.SliderRow
import com.iris.localai.ui.components.SubpageScaffold
import com.iris.localai.ui.components.SwitchRow
import com.iris.localai.ui.components.irisTextFieldColors
import com.iris.localai.ui.feedback.FeedbackEvent
import com.iris.localai.ui.feedback.rememberIrisFeedback
import com.iris.localai.ui.icons.IrisIcons
import com.iris.localai.ui.theme.Iris
import com.iris.localai.ui.theme.IrisDimens
import com.iris.localai.ui.theme.IrisRadius
import com.iris.localai.ui.theme.IrisType
import kotlinx.coroutines.launch
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.IOException
import java.io.File
import java.util.UUID

/** 3.0 AI 工具箱：功能方块网格（每两格一行）。加新功能 = 在 tools 列表里加一项，需要独立页的再在 when(open) 里接上。 */
private data class Tool(val id: String, val icon: ImageVector, val title: String, val sub: String)

@Composable
fun ToolboxPage(chat: ChatViewModel, onBack: () -> Unit, onOpenImageGen: () -> Unit) {
    val ctx = LocalContext.current
    var open by rememberSaveable { mutableStateOf<String?>(null) }
    androidx.activity.compose.BackHandler(open != null) { open = null }
    val floating by FloatingService.running.collectAsStateWithLifecycle()
    when (open) {
        "music" -> SubpageScaffold("音樂生成", { open = null }) { ScrollBody { MusicGenSection() } }
        "video" -> SubpageScaffold("影片生成", { open = null }) { ScrollBody { VideoStudioPage() } }
        "tts" -> SubpageScaffold(I18n.s(R.string.v30_tb_tts), { open = null }) { ScrollBody { TtsSection(chat) } }
        "stt" -> SubpageScaffold(I18n.s(R.string.v30_tb_stt), { open = null }) { ScrollBody { SttSection(chat) } }
        "find" -> SubpageScaffold(I18n.s(R.string.v30_tb_find), { open = null }) { ScrollBody { FindSection() } }
        "float" -> SubpageScaffold(I18n.s(R.string.v30_floating), { open = null }) { ScrollBody { FloatSection(chat, floating) } }
        else -> {
            val tools = listOf(
                Tool("image", IrisIcons.Image, I18n.s(R.string.v30_w_image), I18n.s(R.string.v30_tb_image_sub)),
                Tool("music", IrisIcons.MusicNote, "音樂生成", "輸入描述詞，離線創作專屬音樂"),
                Tool("video", IrisIcons.Movie, "影片生成", "選張圖片，讓畫面動起來"),
                Tool("float", IrisIcons.Layers, I18n.s(R.string.v30_floating), if (floating) I18n.s(R.string.v30_tb_on) else I18n.s(R.string.v30_tb_float_sub2)),
                Tool("tts", IrisIcons.Sound, I18n.s(R.string.v30_tb_tts), I18n.s(R.string.v30_tb_tts_sub)),
                Tool("stt", IrisIcons.Chat, I18n.s(R.string.v30_tb_stt), I18n.s(R.string.v30_tb_stt_sub)),
                Tool("find", IrisIcons.Search, I18n.s(R.string.v30_tb_find), I18n.s(R.string.v30_tb_find_sub)),
            )
            fun tap(id: String) = when (id) {
                "image" -> onOpenImageGen()
                "music", "video" -> { open = id }
                else -> { open = id }
            }
            SubpageScaffold(I18n.s(R.string.v30_toolbox), onBack) {
                ScrollBody {
                    Column(Modifier.fillMaxWidth().padding(horizontal = IrisDimens.groupGutter), verticalArrangement = Arrangement.spacedBy(IrisDimens.space12)) {
                        tools.chunked(2).forEach { row ->
                            Row(Modifier.fillMaxWidth().height(IntrinsicSize.Min), horizontalArrangement = Arrangement.spacedBy(IrisDimens.space12)) {
                                row.forEach { t -> ToolTile(t.icon, t.title, t.sub, Modifier.weight(1f).fillMaxHeight()) { tap(t.id) } }
                                if (row.size == 1) Spacer(Modifier.weight(1f))
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun MusicGenSection() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var bundles by remember { mutableStateOf<List<OnnxBundle>>(emptyList()) }
    var selectedId by rememberSaveable { mutableStateOf<String?>(null) }
    var requiredFilesExpanded by rememberSaveable { mutableStateOf(false) }
    var prompt by rememberSaveable { mutableStateOf("") }
    var duration by rememberSaveable { mutableFloatStateOf(5f) }
    var progress by androidx.compose.runtime.mutableIntStateOf(0)
    var output by remember { mutableStateOf<File?>(null) }
    var player by remember { mutableStateOf<MediaPlayer?>(null) }
    var playerReady by remember { mutableStateOf(false) }
    var playing by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    var statusMessage by remember { mutableStateOf<String?>(null) }
    val clipboard = LocalClipboardManager.current
    val mainHandler = remember { Handler(Looper.getMainLooper()) }

    suspend fun reload() {
        val loaded = withContext(Dispatchers.IO) { OnnxBundleLibrary.list(context.filesDir) }
        bundles = loaded
        val supported = loaded.filter { it.supportsMusicGeneration() }
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

    DisposableEffect(Unit) {
        onDispose {
            player?.release()
            player = null
        }
    }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocumentTree()) { uri ->
        if (uri != null) scope.launch {
            busy = true
            error = null
            statusMessage = null
            try {
                val imported = withContext(Dispatchers.IO) { OnnxBundleLibrary.importTree(context, uri) }
                reload()
                if (imported.supportsMusicGeneration()) {
                    selectedId = imported.id
                } else {
                    error = "這個資料夾還不能用來創作音樂，請查看上方所需檔案清單。"
                }
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                error = e.localizedMessage ?: e.javaClass.simpleName
            } finally {
                busy = false
            }
        }
    }
    val savePicker = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri ->
        val wav = output
        if (uri != null && wav != null) scope.launch {
            try {
                withContext(Dispatchers.IO) {
                    val destination = context.contentResolver.openOutputStream(uri)
                        ?: throw IOException("無法建立 WAV 檔案")
                    wav.inputStream().use { input -> destination.use { output -> input.copyTo(output) } }
                }
                statusMessage = "已儲存到你選擇的位置"
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                error = e.localizedMessage ?: e.javaClass.simpleName
            }
        }
    }
    val musicBundles = bundles.filter { it.supportsMusicGeneration() }
    val selected = musicBundles.firstOrNull { it.id == selectedId }

    SectionTitle("音樂創作工作室")
    SettingsGroup {
        Column(Modifier.fillMaxWidth().padding(IrisDimens.space16), verticalArrangement = Arrangement.spacedBy(IrisDimens.space10)) {
            StepHeading("1", "先載入音樂模型")
            ModelStatusBadge(
                ready = selected != null,
                readyText = "✅ 模型已就緒",
                missingText = "⚠️ 尚未載入音樂模型",
            )
            selected?.let { Text("目前使用：${it.name}", style = IrisType.secondary, color = Iris.colors.textSecondary) }
            IrisButton(
                "選擇模型資料夾",
                onClick = { picker.launch(null) },
                enabled = !busy,
                modifier = Modifier.fillMaxWidth(),
                style = IrisButtonStyle.Primary,
                icon = IrisIcons.Folder,
            )
            if (musicBundles.isNotEmpty()) {
                Text("已匯入的模型", style = IrisType.caption, color = Iris.colors.textSecondary)
                musicBundles.forEach { bundle ->
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(IrisDimens.space8)) {
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
                    MUSIC_REQUIRED_FILES.forEach { Text("• $it", style = IrisType.caption, color = Iris.colors.textSecondary) }
                }
            }
            IrisButton(
                "複製推薦模型名稱（Xenova/musicgen-small）",
                onClick = {
                    clipboard.setText(AnnotatedString(RECOMMENDED_MUSIC_MODEL))
                    statusMessage = "已複製模型名稱，可貼上搜尋"
                },
                compact = true,
                style = IrisButtonStyle.Tonal,
                icon = IrisIcons.Copy,
            )
        }
    }

    SettingsGroup {
        Column(Modifier.fillMaxWidth().padding(IrisDimens.space16), verticalArrangement = Arrangement.spacedBy(IrisDimens.space10)) {
            StepHeading("2", "寫下你想聽的音樂")
            Text("描述心情、樂器或場景，模型會在你的裝置上創作。", style = IrisType.caption, color = Iris.colors.textSecondary)
            OutlinedTextField(
                value = prompt,
                onValueChange = { prompt = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("你想聽到什麼？") },
                placeholder = { Text("例如：雨天窗邊的柔和鋼琴與弦樂") },
                minLines = 3,
                maxLines = 5,
                colors = irisTextFieldColors(),
                textStyle = IrisType.body.copy(color = Iris.colors.text),
            )
            Text("一鍵加入靈感", style = IrisType.caption, color = Iris.colors.textSecondary)
            Row(horizontalArrangement = Arrangement.spacedBy(IrisDimens.space6)) {
                AssistChip(
                    onClick = { prompt = "療癒鋼琴曲，溫柔、舒緩，適合放鬆" },
                    label = { Text("療癒鋼琴曲") },
                )
                AssistChip(
                    onClick = { prompt = "賽博龐克電音，霓虹城市感，節奏鮮明" },
                    label = { Text("賽博龐克電音") },
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(IrisDimens.space6)) {
                AssistChip(
                    onClick = { prompt = "Lo-Fi 讀書節拍，溫暖黑膠質感，安靜專注" },
                    label = { Text("Lo-Fi 讀書節拍") },
                )
                AssistChip(
                    onClick = { prompt = "史詩電影配樂，宏大弦樂與銅管，充滿希望" },
                    label = { Text("史詩電影配樂") },
                )
            }
            SliderRow("音樂長度", duration, 5f..20f, "${duration.toInt()} 秒", subtitle = "目前最長可生成 20 秒", steps = 14) {
                duration = it
            }
            IrisButton(
                when {
                    busy -> "正在創作音樂…"
                    selected == null -> "請先選擇模型資料夾"
                    prompt.isBlank() -> "請輸入音樂描述"
                    else -> "開始創作音樂"
                },
                onClick = {
                    val bundle = selected
                    if (bundle != null && prompt.isNotBlank()) scope.launch {
                        busy = true
                        progress = 0
                        error = null
                        statusMessage = null
                        output = null
                        player?.release()
                        player = null
                        playerReady = false
                        playing = false
                        try {
                            val text = prompt.trim()
                            val seconds = duration.toInt()
                            val wav = withContext(Dispatchers.IO) {
                                val file = File(context.filesDir, "musicgen/${UUID.randomUUID()}.wav")
                                MusicGenOnnxEngine.generate(
                                    context.filesDir,
                                    bundle,
                                    text,
                                    file,
                                    durationSeconds = seconds,
                                    onProgress = { value -> mainHandler.post { progress = value } },
                                ).wav
                            }
                            output = wav
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
                enabled = !busy && selected != null && prompt.isNotBlank(),
                modifier = Modifier.fillMaxWidth(),
                icon = IrisIcons.MusicNote,
            )
            if (selected == null) {
                Text("請先完成上方第 1 步：選擇模型資料夾。", style = IrisType.caption, color = Iris.colors.textTertiary)
            }
        }
    }

    SettingsGroup {
        Column(Modifier.fillMaxWidth().padding(IrisDimens.space16), verticalArrangement = Arrangement.spacedBy(IrisDimens.space10)) {
            StepHeading("3", "試聽與匯出作品")
            Text(
                when {
                    busy -> "正在為你創作… $progress%"
                    output != null -> "創作完成 · 100%"
                    else -> "完成後，作品會出現在這裡"
                },
                style = IrisType.secondary,
                color = if (output != null) Iris.colors.success else Iris.colors.textSecondary,
            )
            LinearProgressIndicator(
                progress = { if (output != null) 1f else (progress / 100f).coerceIn(0f, 1f) },
                modifier = Modifier.fillMaxWidth(),
            )
            output?.let { wav ->
                Text(wav.name, style = IrisType.caption, color = Iris.colors.textTertiary, maxLines = 1, overflow = TextOverflow.Ellipsis)
                IrisButton(
                    when {
                        playing -> "暫停試聽"
                        player != null && !playerReady -> "正在載入音樂…"
                        else -> "播放作品"
                    },
                    onClick = {
                        try {
                            val current = player
                            if (current?.isPlaying == true) {
                                current.pause()
                                playing = false
                            } else if (current != null) {
                                if (playerReady) {
                                    current.seekTo(0)
                                    current.start()
                                    playing = true
                                }
                            } else {
                                val next = MediaPlayer()
                                next.setOnErrorListener { failed, what, extra ->
                                    failed.release()
                                    if (player === failed) player = null
                                    playerReady = false
                                    playing = false
                                    error = "音樂播放失敗（$what/$extra）"
                                    true
                                }
                                next.setOnCompletionListener { playing = false }
                                next.setOnPreparedListener {
                                    playerReady = true
                                    it.start()
                                    playing = true
                                }
                                next.setDataSource(wav.absolutePath)
                                next.prepareAsync()
                                player = next
                            }
                        } catch (e: Exception) {
                            error = e.localizedMessage ?: e.javaClass.simpleName
                        }
                    },
                    enabled = !busy && (player == null || playerReady || playing),
                    modifier = Modifier.fillMaxWidth(),
                    style = IrisButtonStyle.Tonal,
                    icon = IrisIcons.Sound,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(IrisDimens.space8)) {
                    IrisButton(
                        "分享 WAV",
                        onClick = {
                            try {
                                shareWav(context, wav)
                            } catch (e: Exception) {
                                error = e.localizedMessage ?: e.javaClass.simpleName
                            }
                        },
                        modifier = Modifier.weight(1f),
                        compact = true,
                        style = IrisButtonStyle.Outline,
                        icon = IrisIcons.Share,
                    )
                    IrisButton(
                        "儲存 WAV",
                        onClick = { savePicker.launch(wav.name) },
                        modifier = Modifier.weight(1f),
                        compact = true,
                        style = IrisButtonStyle.Outline,
                        icon = IrisIcons.Download,
                    )
                }
            }
        }
    }
    statusMessage?.let { Text(it, style = IrisType.caption, color = Iris.colors.success) }
    error?.let { Text("發生問題：$it", style = IrisType.caption, color = Iris.colors.warning) }
    Spacer(Modifier.height(IrisDimens.space12))
}

private val MUSIC_REQUIRED_FILES = listOf(
    "config.json",
    "tokenizer.json",
    "text_encoder.onnx",
    "decoder_model_merged.onnx",
    "encodec_decode.onnx",
)
private const val RECOMMENDED_MUSIC_MODEL = "Xenova/musicgen-small"

private fun OnnxBundle.supportsMusicGeneration(): Boolean {
    val names = assets.map { it.path.substringAfterLast('/') }.toSet()
    return names.containsAll(MUSIC_REQUIRED_FILES)
}

@Composable
private fun StepHeading(number: String, title: String) {
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
private fun ModelStatusBadge(ready: Boolean, readyText: String, missingText: String) {
    val color = if (ready) Iris.colors.success else Iris.colors.warning
    Text(
        if (ready) readyText else missingText,
        style = IrisType.label,
        color = color,
        modifier = Modifier.fillMaxWidth()
            .background(color.copy(alpha = 0.12f), IrisRadius.mediumShape)
            .padding(horizontal = IrisDimens.space12, vertical = IrisDimens.space10),
    )
}

private fun shareWav(context: android.content.Context, wav: File) {
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", wav)
    val share = Intent(Intent.ACTION_SEND).apply {
        type = "audio/wav"
        putExtra(Intent.EXTRA_STREAM, uri)
        clipData = ClipData.newUri(context.contentResolver, wav.name, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(share, "分享音樂"))
}

/** Beta3：悬浮窗设置页（工具箱里点进来，不再一点就开）。 */
@Composable
private fun FloatSection(chat: ChatViewModel, running: Boolean) {
    val ctx = LocalContext.current
    val st = chat.settings.collectAsStateWithLifecycle().value
    // 从系统授权页回来时重新读权限
    var tick by remember { mutableStateOf(0) }
    val owner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    androidx.compose.runtime.DisposableEffect(owner) {
        val o = androidx.lifecycle.LifecycleEventObserver { _, e -> if (e == androidx.lifecycle.Lifecycle.Event.ON_RESUME) tick++ }
        owner.lifecycle.addObserver(o); onDispose { owner.lifecycle.removeObserver(o) }
    }
    val canDraw = remember(tick) { FloatingService.canDraw(ctx) }
    val a11y = remember(tick) { com.iris.localai.agent.gui.android.IrisAccessibilityService.connected }
    Spacer(Modifier.height(IrisDimens.space8))
    SettingsGroup {
        SwitchRow(I18n.s(R.string.v31_fl_set_on), running, subtitle = if (canDraw) null else I18n.s(R.string.v31_fl_perm), icon = IrisIcons.Layers) { on ->
            when {
                !on -> FloatingService.stop(ctx)
                !FloatingService.canDraw(ctx) -> runCatching { ctx.startActivity(FloatingService.permissionIntent(ctx)) }
                else -> FloatingService.start(ctx)
            }
        }
        SwitchRow(I18n.s(R.string.v31_fl_set_peek), st.floatPeek) { on -> chat.updateSettings { it.copy(floatPeek = on) } }
        SliderRow(I18n.s(R.string.v31_fl_set_size), st.floatSize.toFloat(), 40f..64f, "${st.floatSize} dp", steps = 5) { v -> chat.updateSettings { it.copy(floatSize = v.toInt()) } }
    }
    SectionTitle(I18n.s(R.string.v31_fl_set_menu))
    SettingsGroup {
        val names = mapOf(
            com.iris.localai.floating.FloatingLogic.Mode.TRANSLATE to I18n.s(R.string.v31_fl_tr_screen),
            com.iris.localai.floating.FloatingLogic.Mode.SCREEN to I18n.s(R.string.v30_fl_screen),
            com.iris.localai.floating.FloatingLogic.Mode.ASK to I18n.s(R.string.v30_fl_ask),
        )
        com.iris.localai.floating.FloatingLogic.MENU.forEach { m ->
            SwitchRow(names.getValue(m), m.name in st.floatItems) { on ->
                chat.updateSettings { it.copy(floatItems = if (on) it.floatItems + m.name else it.floatItems - m.name) }
            }
        }
    }
    if (!a11y) {
        Spacer(Modifier.height(IrisDimens.space12))
        SettingsGroup {
            SettingsRow(I18n.s(R.string.v31_fl_a11y), onClick = {
                runCatching { ctx.startActivity(android.content.Intent(android.provider.Settings.ACTION_ACCESSIBILITY_SETTINGS).addFlags(android.content.Intent.FLAG_ACTIVITY_NEW_TASK)) }
            })
        }
    }
}

@Composable
private fun ToolTile(icon: ImageVector, title: String, sub: String, modifier: Modifier, onClick: () -> Unit) {
    val c = Iris.colors
    IrisCard(modifier.clickable(onClick = onClick)) {
        Column(Modifier.fillMaxWidth().padding(IrisDimens.space16)) {
            Box(Modifier.size(40.dp).background(c.accent.copy(alpha = 0.12f), IrisRadius.mediumShape), contentAlignment = Alignment.Center) {
                Icon(icon, null, tint = c.accent, modifier = Modifier.size(22.dp))
            }
            Spacer(Modifier.height(IrisDimens.space12))
            Text(title, style = IrisType.label, color = c.text)
            Text(sub, style = IrisType.caption, color = c.textSecondary, maxLines = 2, overflow = TextOverflow.Ellipsis)
        }
    }
}

@Composable
private fun TtsSection(chat: ChatViewModel) {
    val c = Iris.colors
    var text by rememberSaveable { mutableStateOf("") }
    val speaking by chat.speakingMessageId.collectAsStateWithLifecycle()
    SectionTitle(I18n.s(R.string.v30_tb_tts))
    SettingsGroup {
        Column(Modifier.padding(IrisDimens.space16)) {
            OutlinedTextField(
                text, { text = it }, colors = irisTextFieldColors(), textStyle = IrisType.body.copy(color = c.text),
                placeholder = { Text(I18n.s(R.string.v30_tb_tts_hint), color = c.textTertiary) },
                modifier = Modifier.fillMaxWidth().heightIn(min = 96.dp, max = 220.dp),
            )
            Spacer(Modifier.height(IrisDimens.space12))
            Row(horizontalArrangement = Arrangement.spacedBy(IrisDimens.space8)) {
                if (speaking == TB_SPEAK_ID) IrisButton(I18n.s(R.string.v30_island_stop), { chat.stopSpeaking() }, compact = true, style = IrisButtonStyle.Tonal)
                else IrisButton(I18n.s(R.string.v30_tb_speak), { chat.speakText(text, TB_SPEAK_ID) }, compact = true, enabled = text.isNotBlank(), icon = IrisIcons.Sound)
            }
        }
    }
}

private const val TB_SPEAK_ID = "toolbox-tts"

@Composable
private fun SttSection(chat: ChatViewModel) {
    val c = Iris.colors
    val clipboard = LocalClipboardManager.current
    val scope = rememberCoroutineScope()
    val haptic = rememberIrisFeedback()
    var result by rememberSaveable { mutableStateOf("") }
    var holding by remember { mutableStateOf(false) }
    var busy by remember { mutableStateOf(false) }
    val st = chat.settings.collectAsStateWithLifecycle().value
    val lang = st.asrLanguage
    // 3.0.0：第一次在工具箱里用语音转文本时先申请麦克风权限（以前直接 start，没权限就什么都不发生）
    val ctx = LocalContext.current
    var micDenied by remember { mutableStateOf(false) }
    val micLauncher = androidx.activity.compose.rememberLauncherForActivityResult(androidx.activity.result.contract.ActivityResultContracts.RequestPermission()) { ok -> micDenied = !ok }
    // Beta2：可选识别模型——列出已装的语音识别包，点选即设为当前（与 模型 页 的选择同一设置）
    val packStates by chat.voiceStore.states.collectAsStateWithLifecycle()
    val asrPacks = remember(packStates) {
        com.iris.localai.voice.VoicePack.entries.filter { it.kind == com.iris.localai.voice.VoiceKind.ASR && packStates[it] == com.iris.localai.voice.PackState.Installed }
    }
    val activeAsr = remember(packStates, st.asrPack) { chat.voiceStore.active(com.iris.localai.voice.VoiceKind.ASR, st.asrPack) }
    SectionTitle(I18n.s(R.string.v30_tb_stt))
    SettingsGroup {
        Column(Modifier.padding(IrisDimens.space16)) {
            Text(I18n.s(R.string.v30b2_stt_model), style = IrisType.caption, color = c.textSecondary)
            Spacer(Modifier.height(IrisDimens.space8))
            if (asrPacks.isEmpty()) Text(I18n.s(R.string.v30b2_stt_none), style = IrisType.body, color = c.textTertiary)
            else Row(Modifier.horizontalScroll(androidx.compose.foundation.rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(IrisDimens.space8)) {
                asrPacks.forEach { p ->
                    val sel = p == activeAsr
                    Box(
                        Modifier.clip(IrisRadius.pillShape)
                            .background(com.iris.localai.ui.motion.selectTint(if (sel) c.accentContainer else c.surfaceVariant))
                            .clickable(enabled = !holding && !busy) { chat.useAsrPack(p.id) }
                            .padding(horizontal = IrisDimens.space12, vertical = IrisDimens.space6),
                    ) { Text(p.title, style = IrisType.labelSmall, color = com.iris.localai.ui.motion.selectTint(if (sel) c.onAccentContainer else c.textSecondary)) }
                }
            }
            Spacer(Modifier.height(IrisDimens.space12))
            Box(
                Modifier.fillMaxWidth().height(52.dp)
                    .background(com.iris.localai.ui.motion.selectTint(if (holding) c.accent else c.surfaceVariant, "pttBg"), IrisRadius.pillShape)
                    .pointerInput(lang) {
                        detectTapGestures(onPress = {
                            if (busy) return@detectTapGestures
                            if (ctx.checkSelfPermission(android.Manifest.permission.RECORD_AUDIO) != android.content.pm.PackageManager.PERMISSION_GRANTED) {
                                micLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
                                return@detectTapGestures
                            }
                            micDenied = false
                            holding = true; haptic.perform(FeedbackEvent.LISTEN_START)
                            chat.dictation.warmLanguage = lang
                            chat.dictation.start()
                            tryAwaitRelease()
                            holding = false; haptic.perform(FeedbackEvent.LISTEN_STOP)
                            busy = true
                            scope.launch {
                                val t = runCatching { chat.dictation.finish(lang) }.getOrDefault("")
                                result = com.iris.localai.voice.DictationText.join(result, t)
                                busy = false
                            }
                        })
                    },
                contentAlignment = Alignment.Center,
            ) {
                val reduce = LocalReduceMotion.current
                val fg = com.iris.localai.ui.motion.selectTint(if (holding) c.onAccent else c.text, "pttFg")
                AnimatedContent(
                    when { holding -> I18n.s(R.string.v30_ptt_release); busy -> I18n.s(R.string.v30_tb_recognizing); else -> I18n.s(R.string.v30_ptt_hold) },
                    transitionSpec = { motion(reduce) { IrisMotion.crossfade() } }, label = "ptt",
                ) { t -> Text(t, style = IrisType.label, color = fg) }
            }
            if (micDenied) {
                Spacer(Modifier.height(IrisDimens.space8))
                Text(I18n.s(R.string.b30_mic_denied), style = IrisType.caption, color = c.danger)
            }
            // 清空时收起，退场期间仍显示最后一段
            val shownResult = rememberLast(result.ifBlank { null }).orEmpty()
            Fold(result.isNotBlank()) {
                Column {
                    Spacer(Modifier.height(IrisDimens.space12))
                    SelectionContainer { Text(shownResult, style = IrisType.body, color = c.text) }
                    Row(Modifier.padding(top = IrisDimens.space8), horizontalArrangement = Arrangement.spacedBy(IrisDimens.space8)) {
                        IrisButton(I18n.s(R.string.v30_fl_copy), { clipboard.setText(AnnotatedString(shownResult)) }, compact = true, style = IrisButtonStyle.Tonal, icon = IrisIcons.Copy)
                        IrisButton(I18n.s(R.string.v30_tb_to_chat), { chat.setDraft(shownResult) }, compact = true, style = IrisButtonStyle.Tonal, icon = IrisIcons.Chat)
                        IrisButton(I18n.s(R.string.v30_tb_clear), { result = "" }, compact = true, style = IrisButtonStyle.Text)
                    }
                }
            }
        }
    }
}

@Composable
private fun FindSection() {
    val c = Iris.colors
    val ctx = LocalContext.current
    val scope = rememberCoroutineScope()
    var q by rememberSaveable { mutableStateOf("") }
    var hits by remember { mutableStateOf<List<FileFinder.Hit>>(emptyList()) }
    var searching by remember { mutableStateOf(false) }
    var searched by remember { mutableStateOf(false) }
    val owner = LocalLifecycleOwner.current
    // 从系统「全部文件访问」页回来时重新读权限
    var access by remember { mutableStateOf(FileFinder.hasAccess()) }
    DisposableEffect(owner) {
        val o = LifecycleEventObserver { _, e ->
            if (e == Lifecycle.Event.ON_STOP) FileFinder.release()
            if (e == Lifecycle.Event.ON_RESUME) access = FileFinder.hasAccess()
        }
        owner.lifecycle.addObserver(o)
        onDispose { owner.lifecycle.removeObserver(o); FileFinder.release() }
    }
    SectionTitle(I18n.s(R.string.v30_tb_find))
    if (!access) {
        SettingsGroup {
            Column(Modifier.padding(IrisDimens.space16)) {
                Text(I18n.s(R.string.b30_find_perm), style = IrisType.body, color = c.text)
                Spacer(Modifier.height(IrisDimens.space8))
                IrisButton(I18n.s(R.string.b30_find_perm_go), { com.iris.localai.ui.models.openAllFilesAccess(ctx) }, compact = true, icon = IrisIcons.Folder)
            }
        }
        Spacer(Modifier.height(IrisDimens.space12))
    }
    SettingsGroup {
        Column(Modifier.padding(IrisDimens.space16)) {
            Text(
                if (FileFinder.semanticAvailable()) I18n.s(R.string.v30_tb_find_sem) else I18n.s(R.string.v30_tb_find_kw),
                style = IrisType.caption, color = c.textSecondary,
            )
            Spacer(Modifier.height(IrisDimens.space8))
            OutlinedTextField(
                q, { q = it }, singleLine = true, colors = irisTextFieldColors(), textStyle = IrisType.body.copy(color = c.text),
                placeholder = { Text(I18n.s(R.string.v30_tb_find_hint), color = c.textTertiary) },
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(IrisDimens.space8))
            IrisButton(
                if (searching) I18n.s(R.string.v30_tb_searching) else I18n.s(R.string.v30_fl_search),
                { searching = true; scope.launch { hits = runCatching { FileFinder.search(q) }.getOrDefault(emptyList()); searching = false; searched = true } },
                compact = true, enabled = q.isNotBlank() && !searching, icon = IrisIcons.Search,
            )
        }
        if (searched && hits.isEmpty()) SettingsRow(I18n.s(R.string.v30_fl_nothing))
        hits.take(20).forEach { h ->
            SettingsRow(
                h.file.name, subtitle = h.file.parent.orEmpty().substringAfter("/0/"), subtitleMaxLines = 1,
                icon = if (FindScore.kindWords(h.file.extension).startsWith("照片")) IrisIcons.Image else IrisIcons.File,
                onClick = { openFile(ctx, h.file) },
            )
        }
    }
}

private fun openFile(ctx: android.content.Context, f: File) {
    runCatching {
        val uri: Uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".files", f)
        val type = android.webkit.MimeTypeMap.getSingleton().getMimeTypeFromExtension(f.extension.lowercase()) ?: "*/*"
        ctx.startActivity(Intent.createChooser(Intent(Intent.ACTION_VIEW).setDataAndType(uri, type).addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION), f.name).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    }
}
