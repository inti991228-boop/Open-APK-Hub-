package com.iris.localai.musicgen

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.text.Normalizer

/**
 * Viterbi tokenizer for the SentencePiece Unigram model stored by Hugging Face in tokenizer.json.
 * BPE tokenizers are rejected instead of silently producing incompatible token IDs.
 */
class UnigramTokenizer private constructor(
    private val pieces: List<Piece>,
    private val unkId: Int,
    private val bosId: Int,
    private val eosId: Int,
    private val padId: Int,
) {
    private data class Piece(val text: String, val score: Double, val id: Int)
    private data class Node(val children: MutableMap<Char, Node> = HashMap(), var piece: Piece? = null)

    private val root = Node()

    init {
        pieces.forEach { piece ->
            if (piece.text.isEmpty()) return@forEach
            var node = root
            piece.text.forEach { c -> node = node.children.getOrPut(c) { Node() } }
            node.piece = piece
        }
    }

    companion object {
        fun fromTokenizerJson(tokenizerJson: String): UnigramTokenizer {
            val root = Json.parseToJsonElement(tokenizerJson).jsonObject
            val model = root["model"]?.jsonObject
                ?: throw IllegalArgumentException("tokenizer.json is missing model")
            val type = model["type"]?.jsonPrimitive?.contentOrNull
            require(type == "Unigram") {
                "Expected a SentencePiece Unigram tokenizer; found ${type ?: "unknown"}"
            }
            val vocab = model["vocab"]?.jsonArray
                ?: throw IllegalArgumentException("Unigram tokenizer is missing model.vocab")
            val pieces = vocab.mapIndexedNotNull { id, item ->
                val pair = item.jsonArray
                if (pair.size < 2) throw IllegalArgumentException("Invalid Unigram vocabulary entry $id")
                val text = pair[0].jsonPrimitive.contentOrNull
                    ?: throw IllegalArgumentException("Invalid Unigram token at index $id")
                val score = pair[1].jsonPrimitive.doubleOrNull
                    ?: throw IllegalArgumentException("Invalid Unigram score at index $id")
                if (text.isEmpty()) null else Piece(text, score, id)
            }
            require(pieces.isNotEmpty()) { "Unigram vocabulary is empty" }
            val unkId = model["unk_id"]?.jsonPrimitive?.intOrNull ?: -1
            val added = root["added_tokens"]?.jsonArray.orEmpty()
                .associate { token ->
                    token.jsonObject["content"]?.jsonPrimitive?.contentOrNull.orEmpty() to
                        (token.jsonObject["id"]?.jsonPrimitive?.intOrNull ?: -1)
                }
            val vocabIds = pieces.associate { it.text to it.id }
            fun special(vararg candidates: String, fallback: Int): Int =
                candidates.firstNotNullOfOrNull { name ->
                    added[name]?.takeIf { it >= 0 } ?: vocabIds[name]
                } ?: fallback

            return UnigramTokenizer(
                pieces = pieces,
                unkId = unkId,
                bosId = special("<s>", "<bos>", fallback = -1),
                eosId = special("</s>", "<eos>", fallback = special("</s>", fallback = -1)),
                padId = special("<pad>", fallback = 0),
            )
        }
    }

    /** Returns token IDs with the model's EOS appended; no fixed-length padding is added. */
    fun encode(text: String): LongArray {
        val normalized = normalize(text)
        if (normalized.isEmpty()) return longArrayOf(eosId.toLong())
        val best = DoubleArray(normalized.length + 1) { Double.NEGATIVE_INFINITY }
        val previous = IntArray(normalized.length + 1) { -1 }
        val selected = arrayOfNulls<Piece>(normalized.length + 1)
        best[0] = 0.0
        var position = 0
        while (position < normalized.length) {
            val cpLength = Character.charCount(normalized.codePointAt(position))
            if (best[position].isFinite()) {
                var node = root
                var end = position
                while (end < normalized.length) {
                    node = node.children[normalized[end]] ?: break
                    end++
                    node.piece?.let { piece ->
                        val score = best[position] + piece.score
                        if (score > best[end]) {
                            best[end] = score
                            previous[end] = position
                            selected[end] = piece
                        }
                    }
                }
                if (previous[position + cpLength] < 0 && unkId >= 0) {
                    best[position + cpLength] = best[position] - 100f
                    previous[position + cpLength] = position
                }
            }
            position += cpLength
        }

        if (!best.last().isFinite()) {
            throw IllegalArgumentException("Text contains characters not representable by this tokenizer")
        }
        val ids = ArrayList<Long>()
        position = normalized.length
        while (position > 0) {
            val start = previous[position]
            if (start < 0 || start >= position) throw IllegalStateException("Unable to reconstruct Unigram tokenization")
            ids += (selected[position]?.id ?: unkId).toLong()
            position = start
        }
        ids.reverse()
        if (bosId >= 0) ids.add(0, bosId.toLong())
        if (eosId >= 0) ids += eosId.toLong()
        return ids.toLongArray()
    }

    fun padTokenId(): Long = padId.toLong()

    private fun normalize(text: String): String {
        val normalized = Normalizer.normalize(text, Normalizer.Form.NFKC)
            .replace(Regex("\\s+"), " ")
            .trim()
        return if (normalized.isEmpty()) "" else "\u2581" + normalized.replace(" ", "\u2581")
    }
}
