package com.forge.ai

import android.content.Context
import android.util.Base64
import java.security.MessageDigest
import java.security.SecureRandom

/** Device-local owner identity; NOT a server-side license or secure remote authorization. */
class OwnerAccess(context: Context) {
    private val prefs = context.getSharedPreferences("forge_owner", Context.MODE_PRIVATE)
    fun createOrGetKey(): String {
        val existing = prefs.getString("owner_key", null)
        if (existing != null) return existing
        val bytes = ByteArray(32)
        SecureRandom().nextBytes(bytes)
        val key = "FORGE-OWNER-" + Base64.encodeToString(bytes, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
        prefs.edit().putString("owner_key", key).apply()
        return key
    }
    fun validate(key: String): Boolean {
        val expected = prefs.getString("owner_key", null) ?: return false
        return MessageDigest.isEqual(expected.toByteArray(Charsets.UTF_8), key.toByteArray(Charsets.UTF_8))
    }
}
