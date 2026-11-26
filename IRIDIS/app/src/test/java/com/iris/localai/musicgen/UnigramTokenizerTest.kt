package com.iris.localai.musicgen

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class UnigramTokenizerTest {
    @Test
    fun usesHighestScoringSentencePiecePathAndAppendsEos() {
        val tokenizer = UnigramTokenizer.fromTokenizerJson(
            """{"model":{"type":"Unigram","unk_id":2,"vocab":[["<pad>",0.0],["</s>",0.0],["<unk>",-10.0],["▁",-2.0],["h",-2.0],["i",-2.0],["▁hi",-0.1]]},"added_tokens":[{"id":0,"content":"<pad>"},{"id":1,"content":"</s>"}]}""",
        )

        assertArrayEquals(longArrayOf(6L, 1L), tokenizer.encode("hi"))
    }

    @Test
    fun fallsBackToUnknownPieceForUnrepresentedCodePoints() {
        val tokenizer = UnigramTokenizer.fromTokenizerJson(
            """{"model":{"type":"Unigram","unk_id":2,"vocab":[["<pad>",0.0],["</s>",0.0],["<unk>",-10.0],["▁a",-0.1]]},"added_tokens":[{"id":0,"content":"<pad>"},{"id":1,"content":"</s>"}]}""",
        )

        assertArrayEquals(longArrayOf(3L, 2L, 1L), tokenizer.encode("ab"))
    }

    @Test
    fun rejectsBpeTokenizerRatherThanMisTokenizingIt() {
        assertThrows(IllegalArgumentException::class.java) {
            UnigramTokenizer.fromTokenizerJson("""{"model":{"type":"BPE","vocab":{},"merges":[]}}""")
        }
    }
}
