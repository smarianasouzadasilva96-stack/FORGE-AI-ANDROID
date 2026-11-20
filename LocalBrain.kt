package com.forge.ai

import android.content.Context
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.tanh
import kotlin.random.Random

/** Trainable, four-character-context neural language model. Not an instruction-following LLM. */
class LocalBrain(private val context: Context) {
    private val alphabet = (32..126).map(Int::toChar) + listOf('\n', '\t', 'á', 'é', 'í', 'ó', 'ú', 'ã', 'õ', 'ç', 'ê', 'â')
    private val vocab = alphabet.size
    private val width = 48
    private val window = 4
    private val random = Random(System.nanoTime())
    private val embeddings = Array(vocab) { FloatArray(width) { (random.nextFloat() - .5f) * .12f } }
    private val projections = Array(width) { FloatArray(vocab) { (random.nextFloat() - .5f) * .12f } }
    private val biases = FloatArray(vocab)
    private val running = AtomicBoolean(false)
    @Volatile private var trained = false
    private val weightsFile get() = File(context.filesDir, "forge-weights-v2.bin")
    private val mapping = alphabet.withIndex().associate { it.value to it.index }
    private fun encode(c: Char): Int = mapping[c] ?: 0
    fun isTrained(): Boolean = trained
    private fun contextAt(ids: IntArray, position: Int): IntArray = IntArray(window) { k ->
        val index = position - window + k
        if (index < 0) encode(' ') else ids[index]
    }
    private fun forward(inputs: IntArray): Pair<FloatArray, DoubleArray> {
        val hidden = FloatArray(width) { k ->
            tanh((0 until window).sumOf { pos -> embeddings[inputs[pos]][k].toDouble() * (pos + 1) / 10.0 }).toFloat()
        }
        val logits = DoubleArray(vocab) { j -> biases[j].toDouble() + (0 until width).sumOf { k -> hidden[k].toDouble() * projections[k][j] } }
        val maximum = logits.maxOrNull() ?: 0.0
        val probabilities = DoubleArray(vocab) { j -> exp(logits[j] - maximum) }
        val total = probabilities.sum()
        for (j in probabilities.indices) probabilities[j] /= total
        return hidden to probabilities
    }
    fun train(corpus: String, epochs: Int = 4): String {
        if (!running.compareAndSet(false, true)) return "Treinamento já em andamento."
        try {
            require(corpus.length in 100..200_000) { "Use um texto entre 100 e 200.000 caracteres." }
            require(epochs in 1..20) { "Número de épocas inválido." }
            val ids = corpus.map(::encode).toIntArray()
            require(ids.distinct().size >= 4) { "O corpus precisa ter variedade de caracteres." }
            val split = (ids.size * .9).toInt().coerceAtLeast(2)
            var trainLoss = 0.0
            repeat(epochs) {
                trainLoss = 0.0
                for (pos in 1 until split) {
                    val inputs = contextAt(ids, pos)
                    val target = ids[pos]
                    val (hidden, probabilities) = forward(inputs)
                    trainLoss -= ln(probabilities[target].coerceAtLeast(1e-12))
                    val delta = FloatArray(vocab) { j -> (probabilities[j] - if (j == target) 1.0 else 0.0).toFloat() }
                    val hiddenGradient = FloatArray(width) { k ->
                        (0 until vocab).sumOf { j -> delta[j].toDouble() * projections[k][j] }.toFloat() * (1f - hidden[k] * hidden[k])
                    }
                    val rate = .012f
                    for (j in 0 until vocab) {
                        biases[j] -= rate * delta[j].coerceIn(-1f, 1f)
                        for (k in 0 until width) projections[k][j] -= rate * delta[j] * hidden[k]
                    }
                    for (i in 0 until window) {
                        val row = embeddings[inputs[i]]
                        val weight = (i + 1) / 10f
                        for (k in 0 until width) row[k] -= rate * hiddenGradient[k] * weight
                    }
                }
            }
            var validationLoss = 0.0
            for (pos in split until ids.size) validationLoss -= ln(forward(contextAt(ids, pos)).second[ids[pos]].coerceAtLeast(1e-12))
            save()
            trained = true
            return "Treino concluído: ${epochs} épocas, ${ids.size} caracteres. Perda treino: %.3f; validação: %.3f. Modelo de contexto curto, não assistente geral.".format(java.util.Locale.US, trainLoss / (split - 1), validationLoss / (ids.size - split))
        } finally { running.set(false) }
    }
    fun generate(prompt: String, maxChars: Int = 280, temperature: Double = .8): String {
        if (!trained) return "Modelo não treinado. Abra TREINAR para fornecer texto autorizado."
        require(maxChars in 1..2048)
        require(temperature in .1..2.0)
        val ids = prompt.takeLast(window).map(::encode).toMutableList()
        val generated = StringBuilder()
        repeat(maxChars) {
            val inputs = IntArray(window) { i -> ids.getOrNull(ids.size - window + i) ?: encode(' ') }
            val (_, base) = forward(inputs)
            val weights = DoubleArray(vocab) { j -> base[j].coerceAtLeast(1e-12).let { exp(ln(it) / temperature) } }
            var pick = random.nextDouble() * weights.sum()
            var selected = vocab - 1
            for (j in 0 until vocab) { pick -= weights[j]; if (pick <= 0) { selected = j; break } }
            generated.append(alphabet[selected]); ids.add(selected)
            if (ids.size > window) ids.removeAt(0)
        }
        return generated.toString()
    }
    private fun save() {
        val tmp = File(context.filesDir, "forge-weights-v2.tmp")
        DataOutputStream(tmp.outputStream().buffered()).use { out ->
            out.writeInt(0x464F5232); out.writeInt(vocab); out.writeInt(width); out.writeInt(window)
            embeddings.forEach { a -> a.forEach(out::writeFloat) }
            projections.forEach { a -> a.forEach(out::writeFloat) }
            biases.forEach(out::writeFloat)
        }
        if (!tmp.renameTo(weightsFile)) {
            tmp.copyTo(weightsFile, overwrite = true)
            tmp.delete()
        }
    }
    init {
        try {
            val source = if (weightsFile.exists()) weightsFile.inputStream() else context.assets.open("forge-seed-weights.bin")
            DataInputStream(source.buffered()).use { input ->
                require(input.readInt() == 0x464F5232)
                require(input.readInt() == vocab && input.readInt() == width && input.readInt() == window)
                embeddings.forEach { a -> a.indices.forEach { a[it] = input.readFloat() } }
                projections.forEach { a -> a.indices.forEach { a[it] = input.readFloat() } }
                biases.indices.forEach { biases[it] = input.readFloat() }
                trained = true
            }
        } catch (_: Exception) { trained = false }
    }
}
