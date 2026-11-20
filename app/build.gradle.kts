plugins { id("com.android.application"); id("org.jetbrains.kotlin.android") }
android {
    namespace = "com.forge.ai"
    compileSdk = 35
    defaultConfig { applicationId = "com.forge.ai"; minSdk = 26; targetSdk = 35; versionCode = 10; versionName = "1.0-dev" }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}
