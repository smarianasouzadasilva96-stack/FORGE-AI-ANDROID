package com.forge.ai

/** Offline, explicitly rule-based programming examples. Not a language model. */
object CodeLibrary {
    data class Example(val name: String, val language: String, val source: String)
    val examples = listOf(
        Example("Python • CLI", "python", """def main():
    name = input('Seu nome: ').strip()
    print(f'Olá, {name or "mundo"}!')

if __name__ == '__main__':
    main()"""),
        Example("Python • JSON", "python", """import json
from pathlib import Path

def load_json(path: str) -> dict:
    file = Path(path)
    if not file.exists():
        return {}
    return json.loads(file.read_text(encoding='utf-8'))

print(load_json('config.json'))"""),
        Example("JavaScript • Fetch", "javascript", """async function loadData(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${'$'}{response.status}`);
  return response.json();
}
loadData('/data.json').then(console.log).catch(console.error);"""),
        Example("HTML • Página", "html", """<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Meu projeto</title>
<style>body{font:16px system-ui;max-width:760px;margin:3rem auto;padding:1rem}</style>
</head><body><h1>Olá, mundo!</h1><p>Seu projeto começa aqui.</p></body></html>"""),
        Example("Kotlin • Função", "kotlin", """fun greet(name: String): String {
    val safeName = name.trim().ifEmpty { "mundo" }
    return "Olá, ${'$'}safeName!"
}
fun main() { println(greet("FORGE")) }"""),
        Example("Lua • Tabela", "lua", """local inventory = { coins = 0 }
local function addCoins(amount)
    assert(type(amount) == 'number' and amount >= 0)
    inventory.coins = inventory.coins + amount
end
addCoins(10)
print(inventory.coins)""")
    )
    fun find(prompt: String): Example? {
        val p = prompt.lowercase()
        return when {
            "python" in p && ("json" in p || "arquivo" in p) -> examples[1]
            "python" in p -> examples[0]
            "javascript" in p || "fetch" in p -> examples[2]
            "html" in p || "site" in p -> examples[3]
            "kotlin" in p || "android" in p -> examples[4]
            "lua" in p || "luau" in p -> examples[5]
            else -> null
        }
    }
}
