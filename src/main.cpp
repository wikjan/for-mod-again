#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/modify/PauseLayer.hpp>
#include <Geode/ui/Notification.hpp>
#include <vector>
#include <algorithm>

using namespace geode::prelude;

struct FrameDecision {
    bool hold = false;
};

namespace BotState {
    inline bool s_enabled = false;
    inline int s_currentFrame = 0;
    inline std::vector<FrameDecision> s_bestPath;
    inline std::vector<FrameDecision> s_currentAttempt;
    inline float s_bestDistanceX = 0.0f;
    inline int s_attemptsCount = 0;

    // Elementy wizualne Cheat Indicatora
    inline CCDrawNode* s_indicatorDot = nullptr;
    inline CCLabelBMFont* s_indicatorLabel = nullptr;

    void refreshIndicator() {
        if (!s_indicatorDot || !s_indicatorLabel) return;

        bool show = Mod::get()->getSettingValue<bool>("cheat-indicator");
        s_indicatorDot->setVisible(show);
        s_indicatorLabel->setVisible(show);

        s_indicatorDot->clear();
        if (s_enabled) {
            // CZERWONY: Bot przejmuje kontrole (Cheat Indicator aktywny)
            s_indicatorDot->drawDot(ccp(0, 0), 6.5f, ccc4f(0.95f, 0.15f, 0.15f, 1.0f));
            s_indicatorLabel->setString("AI BOT [RED / ACTIVE]");
            s_indicatorLabel->setColor(ccc3(245, 60, 60));
        } else {
            // ZIELONY: Czysta reczna gra gracza (Legit)
            s_indicatorDot->drawDot(ccp(0, 0), 6.5f, ccc4f(0.15f, 0.95f, 0.25f, 1.0f));
            s_indicatorLabel->setString("LEGIT [GREEN]");
            s_indicatorLabel->setColor(ccc3(50, 240, 80));
        }
    }

    void toggle() {
        s_enabled = !s_enabled;
        Mod::get()->setSettingValue("bot-enabled", s_enabled);
        refreshIndicator();
    }
}

// -------------------------------------------------------------
// 1. Hook PlayLayer: Wskaznik, klatki fizyki i symulacja klikniec
// -------------------------------------------------------------
class $modify(AIBotPlayLayer, PlayLayer) {
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;

        BotState::s_enabled = Mod::get()->getSettingValue<bool>("bot-enabled");
        BotState::s_currentFrame = 0;
        BotState::s_currentAttempt = BotState::s_bestPath;

        // Tworzenie Cheat Indicatora na ekranie gry
        auto winSize = CCDirector::sharedDirector()->getWinSize();
        auto hudNode = CCNode::create();
        hudNode->setPosition(ccp(22.0f, winSize.height - 24.0f));
        hudNode->setZOrder(1000);
        hudNode->setID("ai-cheat-indicator-hud"_spr);

        auto dot = CCDrawNode::create();
        hudNode->addChild(dot);
        BotState::s_indicatorDot = dot;

        auto label = CCLabelBMFont::create("LEGIT [GREEN]", "chatFont.fnt");
        label->setScale(0.55f);
        label->setAnchorPoint(ccp(0.0f, 0.5f));
        label->setPosition(ccp(12.0f, 0.0f));
        hudNode->addChild(label);
        BotState::s_indicatorLabel = label;

        this->addChild(hudNode);
        BotState::refreshIndicator();

        return true;
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        BotState::s_currentFrame = 0;
        BotState::s_currentAttempt = BotState::s_bestPath;
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);

        if (!BotState::s_enabled) return;
        if (!this->m_player1 || this->m_player1->m_isDead) return;

        BotState::s_currentFrame++;

        // Jesli klatka wychodzi poza dotychczasowa pamiec, rozszerzamy sciezke
        if (BotState::s_currentFrame >= (int)BotState::s_currentAttempt.size()) {
            BotState::s_currentAttempt.push_back({false});
        }

        // Zastosowanie zaplanowanego wejscia
        bool hold = BotState::s_currentAttempt[BotState::s_currentFrame].hold;
        if (hold) {
            this->pushButton(0, true);
        } else {
            this->releaseButton(0, true);
        }
    }
};

// -------------------------------------------------------------
// 2. Hook PlayerObject: Reagowanie na smierc i uczenie na bledach
// -------------------------------------------------------------
class $modify(AIBotPlayer, PlayerObject) {
    void playerDestroyed(bool p0) {
        PlayerObject::playerDestroyed(p0);

        if (!BotState::s_enabled) return;

        BotState::s_attemptsCount++;
        float currentX = this->getPositionX();
        int deathFrame = BotState::s_currentFrame;
        int mutationWindow = (int)Mod::get()->getSettingValue<int64_t>("learning-rate");

        // 1. Sprawdzamy czy pobilismy dotychczasowy rekord odleglosci
        if (currentX > BotState::s_bestDistanceX) {
            BotState::s_bestDistanceX = currentX;
            BotState::s_bestPath = BotState::s_currentAttempt;
        }

        // 2. Algorytm uczenia: cofamy sie o N klatek przed moment zderzenia i modyfikujemy decyzje
        int mutateFrom = std::max(0, deathFrame - mutationWindow);
        for (int f = mutateFrom; f <= deathFrame; ++f) {
            if (f < (int)BotState::s_bestPath.size()) {
                // Mutacja akcji w klatce bledu (zamiana skoku / odpuszczenia)
                BotState::s_bestPath[f].hold = !BotState::s_bestPath[f].hold;
            }
        }

        // Jesli wlaczony szybki restart
        if (Mod::get()->getSettingValue<bool>("auto-reset")) {
            auto pl = PlayLayer::get();
            if (pl) {
                pl->resetLevel();
            }
        }
    }
};

// -------------------------------------------------------------
// 3. Hook PauseLayer: Przycisk przelaczania i zarzadzanie w menu pauzy
// -------------------------------------------------------------
class $modify(AIBotPauseLayer, PauseLayer) {
    void customSetup() {
        PauseLayer::customSetup();

        auto menu = this->getChildByID("right-button-menu");
        if (!menu) menu = this->getChildByID("bottom-button-menu");
        if (!menu) return;

        // Przycisk bota w menu pauzy
        auto spr = CircleButtonSprite::create(
            CCLabelBMFont::create("AI", "bigFont.fnt"),
            BotState::s_enabled ? CircleBaseColor::Green : CircleBaseColor::Gray,
            CircleBaseSize::Small
        );

        auto btn = CCMenuItemSpriteExtra::create(
            spr,
            this,
            menu_selector(AIBotPauseLayer::onToggleAI)
        );
        btn->setID("ai-bot-pause-toggle"_spr);

        menu->addChild(btn);
        menu->updateLayout();
    }

    void onToggleAI(CCObject*) {
        BotState::toggle();

        Notification::create(
            BotState::s_enabled ? "AI Bot: WLACZONY (Cheat Indicator: CZERWONY)" : "AI Bot: WYLACZONY (Cheat Indicator: ZIELONY)",
            BotState::s_enabled ? NotificationIcon::Warning : NotificationIcon::Success
        )->show();

        // Przeladuj menu pauzy, aby odswiezyc wyglad przycisku
        this->removeAllChildrenWithCleanup(true);
        this->customSetup();
    }
};

// -------------------------------------------------------------
// 4. Rejestracja nasluchiwania zmian w menu modow Geode
// -------------------------------------------------------------
$on_mod(Loaded) {
    Mod::get()->listenForSettingChanges("bot-enabled", [](bool val) {
        BotState::s_enabled = val;
        BotState::refreshIndicator();
    });
    Mod::get()->listenForSettingChanges("cheat-indicator", [](bool val) {
        BotState::refreshIndicator();
    });
}
