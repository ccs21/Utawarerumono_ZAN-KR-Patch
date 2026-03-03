// dinput8_hook.cpp - runtime text swap (JP->KR) via MultiByteToWideChar hook
// Build: uses your b.bat + MinHook
// Map file: <exe_dir>\ko_patch\zan_map.tsv   (UTF-8, "jp<TAB>kr" per line)

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shlwapi.h>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#pragma comment(lib, "Shlwapi.lib")

#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>

#include "MinHook.h"

// ============================================================
// Real system dinput8 loader
// ============================================================
static HMODULE g_real = nullptr;

static bool LoadRealDinput8() {
    if (g_real) return true;
    wchar_t sysdir[MAX_PATH]{};
    if (!GetSystemDirectoryW(sysdir, MAX_PATH)) return false;
    wchar_t path[MAX_PATH]{};
    lstrcpyW(path, sysdir);
    PathAppendW(path, L"dinput8.dll");
    g_real = LoadLibraryW(path);
    return g_real != nullptr;
}

static FARPROC GetRealProc(const char* name) {
    if (!LoadRealDinput8()) return nullptr;
    return GetProcAddress(g_real, name);
}

// ============================================================
// Logger (optional, for sanity)
// ============================================================
static SRWLOCK g_logLock = SRWLOCK_INIT;
static HANDLE  g_logFile = INVALID_HANDLE_VALUE;

static std::wstring GetExeDir() {
    wchar_t p[MAX_PATH]{};
    GetModuleFileNameW(nullptr, p, MAX_PATH);
    PathRemoveFileSpecW(p);
    return p;
}

static void LogOpenOnce() {
    AcquireSRWLockExclusive(&g_logLock);
    if (g_logFile != INVALID_HANDLE_VALUE) { ReleaseSRWLockExclusive(&g_logLock); return; }
    std::wstring file = GetExeDir() + L"\\zan_swap.log";
    g_logFile = CreateFileW(file.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (g_logFile != INVALID_HANDLE_VALUE) {
        SetFilePointer(g_logFile, 0, nullptr, FILE_END);
        LARGE_INTEGER sz{};
        if (GetFileSizeEx(g_logFile, &sz) && sz.QuadPart == 0) {
            DWORD w=0; const unsigned char bom[3]={0xEF,0xBB,0xBF};
            WriteFile(g_logFile, bom, 3, &w, nullptr);
            FlushFileBuffers(g_logFile);
        }
    }
    ReleaseSRWLockExclusive(&g_logLock);
}

static std::string WToUtf8(const std::wstring& ws) {
    if (ws.empty()) return {};
    int need = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(),
                                   nullptr, 0, nullptr, nullptr);
    if (need <= 0) return {};
    std::string out; out.resize((size_t)need);
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(),
                        &out[0], need, nullptr, nullptr);
    return out;
}

static void LogLineW(const std::wstring& msg) {
    LogOpenOnce();
    if (g_logFile == INVALID_HANDLE_VALUE) return;
    std::string u8 = WToUtf8(msg + L"\r\n");
    AcquireSRWLockExclusive(&g_logLock);
    DWORD w=0;
    WriteFile(g_logFile, u8.data(), (DWORD)u8.size(), &w, nullptr);
    FlushFileBuffers(g_logFile);
    ReleaseSRWLockExclusive(&g_logLock);
}

// ============================================================
// UTF-8 <-> UTF-16 helpers (MUST NOT call hooked APIs recursively)
// We'll call raw system APIs via stored function pointers.
// ============================================================
using PFN_MultiByteToWideChar = int (WINAPI*)(UINT,DWORD,LPCCH,int,LPWSTR,int);

static PFN_MultiByteToWideChar Sys_MultiByteToWideChar = nullptr;  // raw system
static PFN_MultiByteToWideChar Real_MultiByteToWideChar = nullptr; // original target for hook

static thread_local bool g_inHook = false;

static std::wstring Utf8ToW(const std::string& s) {
    if (s.empty() || !Sys_MultiByteToWideChar) return L"";
    int need = Sys_MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    if (need <= 0) return L"";
    std::wstring w;
    w.resize((size_t)need);
    Sys_MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], need);
    return w;
}

static bool LooksJapanese(const wchar_t* s, size_t n) {
    if (!s || n == 0) return false;
    for (size_t i=0;i<n;i++) {
        wchar_t c = s[i];
        // Hiragana, Katakana, CJK Unified Ideographs, Halfwidth Katakana
        if ((c >= 0x3040 && c <= 0x30FF) ||
            (c >= 0x4E00 && c <= 0x9FFF) ||
            (c >= 0xFF66 && c <= 0xFF9D))
            return true;
    }
    return false;
}

// ============================================================
// Map (jp -> kr)
// ============================================================
struct WStrHash {
    size_t operator()(const std::wstring& s) const noexcept {
        // FNV-1a 64-bit
        uint64_t h = 1469598103934665603ull;
        for (wchar_t c : s) {
            h ^= (uint16_t)c;
            h *= 1099511628211ull;
        }
        return (size_t)h;
    }
};
static std::unordered_map<std::wstring, std::wstring, WStrHash> g_map;
static bool g_mapLoaded = false;

static void LoadMapOnce() {
    if (g_mapLoaded) return;
    g_mapLoaded = true;

    std::wstring path = GetExeDir() + L"\\ko_patch\\zan_map.tsv";

    HANDLE h = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        LogLineW(L"[MAP] not found: " + path);
        return;
    }

    LARGE_INTEGER sz{};
    if (!GetFileSizeEx(h, &sz) || sz.QuadPart <= 0 || sz.QuadPart > (1ll<<28)) {
        CloseHandle(h);
        LogLineW(L"[MAP] invalid size");
        return;
    }

    std::string buf;
    buf.resize((size_t)sz.QuadPart);
    DWORD read = 0;
    if (!ReadFile(h, buf.empty() ? nullptr : (LPVOID)&buf[0], (DWORD)buf.size(), &read, nullptr) || read != buf.size()) {
        CloseHandle(h);
        LogLineW(L"[MAP] read fail");
        return;
    }
    CloseHandle(h);

    // strip UTF-8 BOM
    if (buf.size() >= 3 && (uint8_t)buf[0]==0xEF && (uint8_t)buf[1]==0xBB && (uint8_t)buf[2]==0xBF) {
        buf.erase(0, 3);
    }

    // split lines
    size_t pos = 0;
    size_t count = 0;
    while (pos < buf.size()) {
        size_t end = buf.find('\n', pos);
        if (end == std::string::npos) end = buf.size();
        size_t len = end - pos;
        if (len > 0 && buf[pos + len - 1] == '\r') len--;

        if (len > 0) {
            std::string line = buf.substr(pos, len);
            size_t tab = line.find('\t');
            if (tab != std::string::npos) {
                std::string a = line.substr(0, tab);
                std::string b = line.substr(tab + 1);
                if (!a.empty() && !b.empty()) {
                    std::wstring wa = Utf8ToW(a);
                    std::wstring wb = Utf8ToW(b);
                    if (!wa.empty() && !wb.empty()) {
                        g_map[wa] = wb; // last wins
                        count++;
                    }
                }
            }
        }

        pos = (end < buf.size()) ? (end + 1) : end;
    }

    LogLineW(L"[MAP] loaded entries=" + std::to_wstring(g_map.size()));
}

// ============================================================
// Hook: MultiByteToWideChar
// - If input string matches map, output translated wide string instead.
// - This is the "autotranslator" swap point.
// ============================================================
static int WINAPI Hook_MultiByteToWideChar(UINT cp, DWORD flags, LPCCH mb, int cb, LPWSTR wc, int cch)
{
    if (!Real_MultiByteToWideChar) return 0;

    // avoid recursion
    if (g_inHook) return Real_MultiByteToWideChar(cp, flags, mb, cb, wc, cch);
    g_inHook = true;

    // lazy-load map
    LoadMapOnce();

    // only attempt swap for plausible encodings
    bool cpOk = (cp == CP_ACP || cp == 0 || cp == 932 || cp == 65001);
    if (!cpOk || !mb || cb == 0 || g_map.empty()) {
        g_inHook = false;
        return Real_MultiByteToWideChar(cp, flags, mb, cb, wc, cch);
    }

    // normalize input length
    int inLen = cb;
    if (inLen < 0) {
        // null-terminated
        inLen = (int)strlen(mb);
    }
    if (inLen <= 0 || inLen > 2048) {
        g_inHook = false;
        return Real_MultiByteToWideChar(cp, flags, mb, cb, wc, cch);
    }

    // Convert input to wide using ORIGINAL (not our own logic)
    int need = Real_MultiByteToWideChar(cp, 0, mb, inLen, nullptr, 0);
    if (need <= 0 || need > 2048) {
        g_inHook = false;
        return Real_MultiByteToWideChar(cp, flags, mb, cb, wc, cch);
    }

    std::wstring wsrc;
    wsrc.resize((size_t)need);
    Real_MultiByteToWideChar(cp, 0, mb, inLen, &wsrc[0], need);

    // quick filter: JP only
    if (!LooksJapanese(wsrc.c_str(), wsrc.size())) {
        g_inHook = false;
        return Real_MultiByteToWideChar(cp, flags, mb, cb, wc, cch);
    }

    auto it = g_map.find(wsrc);
    if (it == g_map.end()) {
        g_inHook = false;
        return Real_MultiByteToWideChar(cp, flags, mb, cb, wc, cch);
    }

    const std::wstring& wdst = it->second;

    // MultiByteToWideChar semantics:
    // - If cb == -1, return count includes null terminator.
    bool includeNull = (cb < 0);
    int outNeed = (int)wdst.size() + (includeNull ? 1 : 0);

    if (!wc || cch == 0) {
        g_inHook = false;
        return outNeed;
    }
    if (cch < outNeed) {
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        g_inHook = false;
        return 0;
    }

    // write translation
    memcpy(wc, wdst.data(), wdst.size() * sizeof(wchar_t));
    if (includeNull) wc[wdst.size()] = L'\0';

    // optional minimal log (comment out if too noisy)
    // LogLineW(L"[REPL] " + wsrc + L" -> " + wdst);

    g_inHook = false;
    return outNeed;
}

// ============================================================
// Install hooks once
// ============================================================
static INIT_ONCE g_once = INIT_ONCE_STATIC_INIT;

static void InstallOne(void* target, void* detour, void** orig) {
    if (!target) return;
    if (MH_CreateHook(target, detour, orig) == MH_OK) {
        MH_EnableHook(target);
    }
}

static BOOL CALLBACK InitOnceCb(PINIT_ONCE, PVOID, PVOID*)
{
    LogOpenOnce();
    LogLineW(L"[INIT] begin");

    if (MH_Initialize() != MH_OK) {
        LogLineW(L"[INIT] MH_Initialize FAIL");
        return TRUE;
    }

    HMODULE k32 = GetModuleHandleW(L"kernel32.dll");
    HMODULE kbase = GetModuleHandleW(L"kernelbase.dll");

    auto gp = [&](const char* n)->void*{
        void* p = (void*)GetProcAddress(k32, n);
        if (!p && kbase) p = (void*)GetProcAddress(kbase, n);
        return p;
    };

    // capture raw system pointer BEFORE hooking (for UTF-8 map parsing)
    Sys_MultiByteToWideChar = (PFN_MultiByteToWideChar)gp("MultiByteToWideChar");

    void* pMB2WC = gp("MultiByteToWideChar");
    InstallOne(pMB2WC, (void*)Hook_MultiByteToWideChar, (void**)&Real_MultiByteToWideChar);

    LogLineW(L"[INIT] hooks installed (MB2WC)");
    return TRUE;
}

static void EnsureHooksInstalled() {
    InitOnceExecuteOnce(&g_once, InitOnceCb, nullptr, nullptr);
}

// ============================================================
// Export (mapped by dinput8.def)
// ============================================================
extern "C" HRESULT WINAPI My_DirectInput8Create(
    HINSTANCE hinst, DWORD dwVersion, REFIID riidltf,
    LPVOID* ppvOut, LPUNKNOWN punkOuter)
{
    EnsureHooksInstalled();

    using Fn = HRESULT (WINAPI*)(HINSTANCE, DWORD, REFIID, LPVOID*, LPUNKNOWN);
    auto fn = (Fn)GetRealProc("DirectInput8Create");
    if (!fn) return E_FAIL;
    return fn(hinst, dwVersion, riidltf, ppvOut, punkOuter);
}

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }