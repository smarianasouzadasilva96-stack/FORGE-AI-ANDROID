package com.forge.ai

import android.content.Context
import java.io.BufferedInputStream
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*
import kotlin.random.Random

/** Native CPU inference for FORGE's own causal Transformer; no model APIs or pretrained weights. */
class TransformerRuntime(context: Context) {
    private data class Tensor(val shape: IntArray, val values: FloatArray) {
        fun at(i: Int, j: Int) = values[i * shape[1] + j]
    }
    private val weights = HashMap<String, Tensor>()
    private val width: Int
    private val layers: Int
    private val heads: Int
    private val contextSize: Int

    init {
        DataInputStream(BufferedInputStream(context.assets.open("forge-transformer.frg"))).use { input ->
            val magic = ByteArray(4); input.readFully(magic)
            require(String(magic, Charsets.US_ASCII) == "FRG1") { "Invalid model header" }
            fun readIntLE(): Int = Integer.reverseBytes(input.readInt())
            width = readIntLE(); layers = readIntLE(); heads = readIntLE(); contextSize = readIntLE()
            require(width in 1..512 && layers in 1..12 && heads in 1..32 && width % heads == 0 && contextSize in 8..2048)
            val count = readIntLE()
            require(count in 1..512)
            repeat(count) {
                val len = java.lang.Short.toUnsignedInt(java.lang.Short.reverseBytes(input.readShort()))
                require(len in 1..200)
                val keyBytes = ByteArray(len); input.readFully(keyBytes)
                val key = String(keyBytes, Charsets.US_ASCII)
                val rank = input.readUnsignedByte()
                require(rank in 1..3)
                val dims = IntArray(rank) { readIntLE().also { d -> require(d in 1..1000000) } }
                val elements = dims.fold(1L) { a, b -> a * b }
                require(elements in 1..5_000_000)
                val bytes = ByteArray(elements.toInt() * 4); input.readFully(bytes)
                val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                weights[key] = Tensor(dims, FloatArray(elements.toInt()) { buffer.float })
            }
        }
        require(weights.containsKey("head.weight"))
    }

    private fun tensor(name: String): Tensor = weights[name] ?: error("Missing tensor: $name")
    private fun linear(x: FloatArray, prefix: String): FloatArray {
        val w = tensor("$prefix.weight")
        val bias = weights["$prefix.bias"]?.values
        val output = FloatArray(w.shape[0])
        for (i in output.indices) {
            var v = bias?.get(i) ?: 0f
            val offset = i * x.size
            for (j in x.indices) v += w.values[offset + j] * x[j]
            output[i] = v
        }
        return output
    }
    private fun norm(x: FloatArray, name: String): FloatArray {
        val scale = tensor("$name.weight").values
        val bias = tensor("$name.bias").values
        val mean = x.sum() / x.size
        var variance = 0.0
        for (v in x) variance += (v - mean).toDouble().pow(2)
        val inv = 1.0 / sqrt(variance / x.size + 1e-5)
        return FloatArray(x.size) { i -> ((x[i] - mean) * inv * scale[i] + bias[i]).toFloat() }
    }
    private fun add(a: FloatArray, b: FloatArray): FloatArray = FloatArray(a.size) { a[it] + b[it] }
    private fun gelu(v: Float): Float {
        val x = v.toDouble()
        return (0.5 * x * (1.0 + tanh(sqrt(2.0 / Math.PI) * (x + 0.044715 * x.pow(3))))).toFloat()
    }
    private fun logits(ids: IntArray): FloatArray {
        val n = ids.size
        val emb = tensor("token.weight")
        val pos = tensor("position.weight")
        var x = Array(n) { t -> FloatArray(width) { j -> emb.at(ids[t], j) + pos.at(t, j) } }
        for (layer in 0 until layers) {
            val base = "blocks.$layer"
            val normalized = Array(n) { norm(x[it], "$base.norm1") }
            val proj = tensor("$base.attn.in_proj_weight")
            val projBias = tensor("$base.attn.in_proj_bias").values
            val qkv = Array(n) { t -> FloatArray(3 * width) { o ->
                var v = projBias[o]
                for (j in 0 until width) v += proj.at(o, j) * normalized[t][j]
                v
            } }
            val headDim = width / heads
            val attn = Array(n) { FloatArray(width) }
            for (h in 0 until heads) for (t in 0 until n) {
                val scores = DoubleArray(t + 1)
                for (s in 0..t) {
                    var dot = 0.0
                    for (j in 0 until headDim) dot += (qkv[t][h * headDim + j] * qkv[s][width + h * headDim + j]).toDouble()
                    scores[s] = dot / sqrt(headDim.toDouble())
                }
                val maxScore = scores.maxOrNull() ?: 0.0
                var denom = 0.0
                for (s in scores.indices) { scores[s] = exp(scores[s] - maxScore); denom += scores[s] }
                for (j in 0 until headDim) {
                    var v = 0.0
                    for (s in 0..t) v += scores[s] / denom * qkv[s][2 * width + h * headDim + j]
                    attn[t][h * headDim + j] = v.toFloat()
                }
            }
            x = Array(n) { t -> add(x[t], linear(attn[t], "$base.attn.out_proj")) }
            x = Array(n) { t ->
                val hidden = linear(norm(x[t], "$base.norm2"), "$base.ff.0")
                add(x[t], linear(FloatArray(hidden.size) { gelu(hidden[it]) }, "$base.ff.2"))
            }
        }
        val final = norm(x.last(), "norm")
        val head = tensor("head.weight")
        return FloatArray(256) { i ->
            var v = 0f
            for (j in 0 until width) v += head.at(i, j) * final[j]
            v
        }
    }
    fun generate(prompt: String, count: Int = 100, temperature: Double = 0.8): String {
        require(count in 1..512 && temperature in 0.1..2.0)
        val tokens = (prompt.toByteArray(Charsets.UTF_8).map { it.toInt() and 255 }).toMutableList()
        if (tokens.isEmpty()) tokens.add(32)
        val generated = ArrayList<Byte>()
        repeat(count) {
            val input = tokens.takeLast(contextSize).toIntArray()
            val values = logits(input)
            val maxLogit = values.maxOrNull() ?: 0f
            val probs = DoubleArray(256) { exp(((values[it] - maxLogit) / temperature).toDouble()) }
            val sum = probs.sum()
            var sample = Random.nextDouble() * sum
            var next = 255
            for (i in probs.indices) { sample -= probs[i]; if (sample <= 0) { next = i; break } }
            tokens.add(next)
            generated.add(next.toByte())
        }
        return generated.toByteArray().toString(Charsets.UTF_8)
    }
}
