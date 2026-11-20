package com.forge.ai

import android.app.Activity
import android.app.AlertDialog
import android.os.Bundle
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.View
import android.widget.*
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private val background = Color.rgb(9, 13, 20)
    private val surface = Color.rgb(21, 28, 39)
    private val primary = Color.rgb(122, 240, 183)
    private val foreground = Color.rgb(236, 244, 245)
    private val secondary = Color.rgb(156, 172, 181)
    private lateinit var brain: LocalBrain
    private var transformer: TransformerRuntime? = null
    private lateinit var messages: LinearLayout
    private lateinit var input: EditText
    private lateinit var scroller: ScrollView
    private val worker = Executors.newSingleThreadExecutor()
    private var history = JSONArray()
    private var busy = false
    private var destroyed = false
    private lateinit var owner: OwnerAccess
    private var maxGeneration = 280
    private var usedChars = 0L
    private fun dp(x: Int) = (x * resources.displayMetrics.density).toInt()
    private fun bg(color: Int, radius: Int = 16) = GradientDrawable().apply { setColor(color); cornerRadius = dp(radius).toFloat() }
    private fun text(value: String, size: Float = 14f, color: Int = foreground) = TextView(this).apply {
        this.text = value; textSize = size; setTextColor(color); setLineSpacing(dp(3).toFloat(), 1f)
    }
    private fun action(label: String, click: () -> Unit) = TextView(this).apply {
        text = label; textSize = 12f; setTextColor(this@MainActivity.background); gravity = Gravity.CENTER
        typeface = Typeface.DEFAULT_BOLD; background = bg(primary, 12)
        setPadding(dp(10), dp(12), dp(10), dp(12)); setOnClickListener { click() }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        brain = LocalBrain(this)
        transformer = try { TransformerRuntime(this) } catch (_: Exception) { null }
        owner = OwnerAccess(this)
        maxGeneration = getPreferences(MODE_PRIVATE).getInt("max_generation", 280)
        usedChars = getPreferences(MODE_PRIVATE).getLong("used_chars", 0L)
        window.statusBarColor = background; window.navigationBarColor = background
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(this@MainActivity.background); setPadding(dp(14), dp(12), dp(14), dp(12)) }
        root.addView(text("◈  FORGE AI  ∞", 23f, primary).apply { typeface = Typeface.DEFAULT_BOLD })
        root.addView(text("TRANSFORMER LOCAL  •  SEM API  •  MODELO PEQUENO", 10f, secondary).apply { setPadding(0, dp(2), 0, dp(12)) })
        val tools = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        listOf("TREINAR" to { showTraining() }, "CÓDIGOS" to { showExamples() }, "EXPORTAR" to { exportHistory() }, "CONTA" to { showAccount() }).forEachIndexed { index, pair ->
            if (index > 0) tools.addView(View(this), LinearLayout.LayoutParams(dp(5), 1))
            tools.addView(action(pair.first, pair.second), LinearLayout.LayoutParams(0, dp(44), 1f))
        }
        root.addView(tools)
        scroller = ScrollView(this).apply { isFillViewport = true }
        messages = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(16), 0, dp(16)) }
        scroller.addView(messages)
        root.addView(scroller, LinearLayout.LayoutParams(-1, 0, 1f))
        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; background = bg(surface); setPadding(dp(10), dp(6), dp(6), dp(6)); gravity = Gravity.CENTER_VERTICAL }
        input = EditText(this).apply { hint = "Pergunte ou peça um exemplo de código..."; setHintTextColor(secondary); setTextColor(this@MainActivity.foreground); textSize = 14f; background = null; maxLines = 4 }
        row.addView(input, LinearLayout.LayoutParams(0, -2, 1f))
        row.addView(action("ENVIAR") { send() })
        root.addView(row)
        setContentView(root)
        restoreHistory()
        if (history.length() == 0) addMessage("FORGE", "Motor Transformer local disponível. CÓDIGOS oferece exemplos verificados; o Transformer ainda não foi treinado para seguir instruções complexas.", false)
    }
    private fun addMessage(who: String, body: String, mine: Boolean, persist: Boolean = true) {
        val card = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; background = bg(if (mine) Color.rgb(25, 51, 48) else surface); setPadding(dp(14), dp(12), dp(14), dp(12)) }
        card.addView(text(who, 11f, if (mine) primary else secondary).apply { typeface = Typeface.DEFAULT_BOLD })
        card.addView(text(body, 14f).apply { setPadding(0, dp(6), 0, 0); setTextIsSelectable(true) })
        card.setOnLongClickListener {
            val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            clipboard.setPrimaryClip(ClipData.newPlainText("FORGE", body))
            Toast.makeText(this, "Texto copiado", Toast.LENGTH_SHORT).show(); true
        }
        messages.addView(card, LinearLayout.LayoutParams(-1, -2).apply { bottomMargin = dp(10) })
        scroller.post { scroller.fullScroll(View.FOCUS_DOWN) }
        if (persist) { history.put(JSONObject().put("who", who).put("body", body.take(20000)).put("mine", mine));
            if (history.length() > 200) {
                val reduced = JSONArray()
                for (i in history.length() - 200 until history.length()) reduced.put(history.getJSONObject(i))
                history = reduced
            }
            saveHistory() }
    }
    private fun saveHistory() { getPreferences(MODE_PRIVATE).edit().putString("history", history.toString()).apply() }
    private fun restoreHistory() {
        val saved = getPreferences(MODE_PRIVATE).getString("history", "[]") ?: "[]"
        try {
            val array = JSONArray(saved)
            for (i in 0 until array.length()) {
                val item = array.getJSONObject(i)
                history.put(item)
                addMessage(item.optString("who"), item.optString("body"), item.optBoolean("mine"), false)
            }
        } catch (_: Exception) { getPreferences(MODE_PRIVATE).edit().remove("history").apply() }
    }
    private fun exportHistory() {
        if (history.length() == 0) {
            Toast.makeText(this, "Nenhuma conversa para exportar", Toast.LENGTH_SHORT).show()
            return
        }
        val result = buildString {
            append("FORGE AI — Histórico local\n\n")
            for (i in 0 until history.length()) {
                val item = history.optJSONObject(i) ?: continue
                append(item.optString("who", "Mensagem"))
                append(":\n")
                append(item.optString("body"))
                append("\n\n")
            }
        }
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_TEXT, result)
            putExtra(Intent.EXTRA_SUBJECT, "Histórico FORGE AI")
        }, "Exportar histórico"))
    }
    private fun clearHistory() {
        AlertDialog.Builder(this).setTitle("Apagar conversa?").setMessage("O histórico local será apagado; os pesos treinados serão mantidos.")
            .setNegativeButton("Cancelar", null).setPositiveButton("Apagar") { _, _ ->
                messages.removeAllViews(); while (history.length() > 0) history.remove(0); saveHistory()
            }.show()
    }
    private fun send() {
        if (busy) { Toast.makeText(this, "Aguarde a operação terminar", Toast.LENGTH_SHORT).show(); return }
        val prompt = input.text.toString().trim().take(20000)
        if (prompt.isEmpty()) return
        input.setText("")
        addMessage("VOCÊ", prompt, true)
        val example = CodeLibrary.find(prompt)
        if (example != null) {
            addMessage("FORGE • EXEMPLO OFFLINE", "Modelo: ${example.name}\n\n${example.source}\n\nEste é um template fixo e editável, não código criado por IA.", false)
        } else if (transformer != null || brain.isTrained()) {
            busy = true
            worker.execute {
                val response = try { (transformer?.generate(prompt, maxGeneration.coerceIn(1, 512)) ?: brain.generate(prompt, maxGeneration)) } catch (e: Exception) { "Falha na geração: ${e.message}" }
                runOnUiThread {
                    if (destroyed || isFinishing) return@runOnUiThread
                    busy = false
                    usedChars += response.length
                    getPreferences(MODE_PRIVATE).edit().putLong("used_chars", usedChars).apply()
                    addMessage("FORGE • TRANSFORMER LOCAL", response, false)
                }
            }
        } else addMessage("FORGE", "Ainda não fui treinada para gerar texto. Toque em TREINAR ou abra CÓDIGOS para acessar exemplos de programação offline.", false)
    }
    private fun showExamples() {
        val labels = CodeLibrary.examples.map { it.name }.toTypedArray()
        AlertDialog.Builder(this).setTitle("Biblioteca de código").setItems(labels) { _, index ->
            val ex = CodeLibrary.examples[index]
            addMessage("FORGE • ${ex.language.uppercase()}", ex.source, false)
        }.setNegativeButton("Fechar", null).show()
    }
    private fun showTraining() {
        if (busy) return
        val editor = EditText(this).apply { hint = "Cole texto autorizado (mínimo 100 caracteres)."; minLines = 5; maxLines = 10; setPadding(dp(16), dp(16), dp(16), dp(16)); setTextColor(Color.BLACK) }
        AlertDialog.Builder(this).setTitle("Treinar rede neural local").setView(editor)
            .setNegativeButton("Cancelar", null).setPositiveButton("Treinar") { _, _ ->
                val corpus = editor.text.toString()
                if (corpus.length !in 100..200_000) { addMessage("SISTEMA", "Forneça de 100 a 200.000 caracteres.", false); return@setPositiveButton }
                busy = true
                addMessage("SISTEMA", "Treinamento iniciado em segundo plano (dados locais)...", false)
                worker.execute {
                    val result = try { brain.train(corpus, 5) } catch (e: Exception) { "Falha: ${e.message}" }
                    runOnUiThread { if (!destroyed && !isFinishing) { busy = false; addMessage("SISTEMA", result, false) } }
                }
            }.show()
    }
    private fun showAccount() {
        val options = arrayOf("Gerar/visualizar chave de proprietário", "Ativar chave neste aparelho", "Limite por geração", "Limpar histórico", "Estatísticas e informações")
        AlertDialog.Builder(this).setTitle("FORGE • Conta e tokens")
            .setItems(options) { _, choice ->
                when (choice) {
                    0 -> {
                        val key = owner.createOrGetKey()
                        AlertDialog.Builder(this).setTitle("Chave de proprietário")
                            .setMessage("Guarde esta chave em local seguro. Ela identifica o proprietário somente nesta instalação. Não compartilhe.\n\n$key")
                            .setPositiveButton("Copiar") { _, _ ->
                                val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                                clipboard.setPrimaryClip(ClipData.newPlainText("FORGE owner key", key))
                            }.setNegativeButton("Fechar", null).show()
                    }
                    1 -> {
                        val field = EditText(this).apply { hint = "Cole sua chave"; setSingleLine(true) }
                        AlertDialog.Builder(this).setTitle("Ativar proprietário")
                            .setView(field).setPositiveButton("Validar") { _, _ ->
                                val ok = owner.validate(field.text.toString().trim())
                                Toast.makeText(this, if (ok) "Proprietário validado neste aparelho" else "Chave inválida", Toast.LENGTH_LONG).show()
                            }.setNegativeButton("Cancelar", null).show()
                    }
                    2 -> {
                        val values = arrayOf("128 caracteres", "280 caracteres", "512 caracteres", "1024 caracteres", "2048 caracteres")
                        val limits = intArrayOf(128,280,512,1024,2048)
                        AlertDialog.Builder(this).setTitle("Máximo por resposta")
                            .setSingleChoiceItems(values, limits.indexOf(maxGeneration)) { dialog, which ->
                                maxGeneration = limits[which]
                                getPreferences(MODE_PRIVATE).edit().putInt("max_generation", maxGeneration).apply()
                                dialog.dismiss()
                            }.setNegativeButton("Cancelar", null).show()
                    }
                    3 -> clearHistory()
                    else -> showAbout()
                }
            }.setNegativeButton("Fechar", null).show()
    }
    private fun showAbout() {
        AlertDialog.Builder(this).setTitle("FORGE AI ∞ 0.7")
            .setMessage("Protótipo neural offline, sem APIs e sem modelos prontos. A rede aprende padrões de até quatro caracteres e NÃO é um assistente avançado. Os exemplos de código são templates.\n\nPlano: OWNER UNLIMITED (sem cota artificial acumulada). Contexto e geração são finitos; máximo por resposta: $maxGeneration caracteres. Caracteres gerados nesta instalação: $usedChars.\n\nA chave é LOCAL e não serve como autenticação segura em servidores. Pressione mensagens para copiar.")
            .setPositiveButton("OK", null).show()
    }
    override fun onDestroy() { destroyed = true; worker.shutdownNow(); super.onDestroy() }
}
