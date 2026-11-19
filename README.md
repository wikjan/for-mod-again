# AI Learning Bot & Cheat Indicator (Geode 2.2 Mod)

Autonomiczny mod do **Geometry Dash 2.2** stworzony w oparciu o **Geode SDK (C++)**, który sam uczy się przechodzić poziomy na własnych błędach, posiada dynamiczny **Cheat Indicator** oraz pełną integrację z menu pauzy i menu modów.

---

### ✨ Główne Funkcje

1. **Uczenie maszynowe na błędach (Trial-and-Error Learning):**
   * Bot rejestruje klatki gry i wykonuje próby przejścia poziomu.
   * W momencie śmierci (`playerDestroyed`) cofa bufor decyzji o ustaloną liczbę klatek (`learning-rate`) i dokonuje mutacji (zmienia czas skoku, odpuszcza przytrzymanie).
   * Pamięta najdalszy osiągnięty dystans `X` i stale optymalizuje trasę.

2. **Cheat Indicator w rogu ekranu:**
   * **🔴 CZERWONY (`AI BOT [RED / ACTIVE]`):** Bot jest włączony i kieruje graczem.
   * **🟢 ZIELONY (`LEGIT [GREEN]`):** Bot jest wyłączony, gracz gra w 100% manualnie.

3. **Przełączanie w menu pauzy (`PauseLayer`):**
   * Po wciśnięciu pauzy w grze pojawia się dedykowany przycisk **"AI"**.
   * Kliknięcie przełącza bota w locie i aktualizuje wskaźnik oraz powiadomienie na ekranie.

4. **Konfiguracja w menu modów Geode:**
   * Włączanie / wyłączanie bota (`bot-enabled`).
   * Włączanie / wyłączanie Cheat Indicatora (`cheat-indicator`).
   * Regulacja okna mutacji klatek (`learning-rate`).
   * Szybki auto-reset po śmierci (`auto-reset`).

---

### 🔨 Jak zbudować mod?

1. Upewnij się, że masz zainstalowane [Geode CLI](https://docs.geode-sdk.org/).
2. Otwórz terminal w folderze modu:
   ```bash
   cd geode-mod
   ```
3. Zbuduj mod poleceniem:
   ```bash
   geode build
   ```
4. Aby zainstalować go bezpośrednio do Geometry Dash:
   ```bash
   geode build --install
   ```
Plik `.geode` zostanie automatycznie skompilowany i przeniesiony do folderu `geode/mods/` w Twojej instalacji Geometry Dash!
