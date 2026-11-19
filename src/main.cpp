#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/modify/PauseLayer.hpp>
#include <Geode/binding/StartPosObject.hpp>
#include <Geode/ui/Notification.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

using namespace geode::prelude;

struct FrameDecision {
    bool hold = false;
};

struct TrailPoint {
    CCPoint position;
    bool click = false;
};

struct LearnedSegment {
    float startPercent = 0.0f;
    float bestPercent = 0.0f;
    std::vector<FrameDecision> bestInputs;
    std::vector<TrailPoint> bestTrajectory;
};

namespace BotState {
    inline bool s_enabled = false;
    inline int s_currentFrame = 0;
    inline std::vector<FrameDecision> s_bestPath;
    inline std::vector<FrameDecision> s_candidatePath;
    inline std::vector<FrameDecision> s_currentAttempt;
    inline float s_bestDistanceX = 0.0f;
    inline int s_attemptsCount = 0;
    inline float s_lastDeathPercent = -1000.0f;
    inline int s_sameSpotDeaths = 0;
    inline int s_score = 0;
    inline CCDrawNode* s_trajectoryNode = nullptr;
    inline CCDrawNode* s_hitboxNode = nullptr;
    inline std::vector<TrailPoint> s_currentTrajectory;
    inline std::vector<TrailPoint> s_bestTrajectory;
    inline bool s_hasLastTrajectoryPoint = false;
    inline bool s_lastHold = false;
    inline float s_cpsCooldown = 0.0f;
    inline CCPoint s_lastTrajectoryPoint = CCPointZero;
    inline LearnedSegment s_segment;
    inline std::vector<FrameDecision> s_segmentCandidate;
    inline std::vector<FrameDecision> s_segmentAttempt;
    inline bool s_trainingSegment = false;
    inline bool s_startPosTraining = false;
    inline bool s_autoPracticeTraining = false;
    inline bool s_autoCheckpointPlaced = false;
    inline float s_checkpointTargetPercent = 0.0f;
    inline int s_noProgressDeaths = 0;
    inline int s_segmentPlaybackFrame = 0;
    inline bool s_segmentPlaybackActive = false;
    inline bool s_segmentPlaybackDone = false;

    // Elementy wizualne Cheat Indicatora
    inline CCDrawNode* s_indicatorDot = nullptr;
    inline CCLabelBMFont* s_indicatorLabel = nullptr;
    inline CCLabelBMFont* s_trajectoryLegend = nullptr;
    inline CCLabelBMFont* s_scoreLabel = nullptr;

    std::vector<FrameDecision> makeCandidate(
        std::vector<FrameDecision> const& champion,
        int deathFrame,
        int mutationWindow,
        uint32_t seed
    ) {
        auto candidate = champion;
        if ((int)candidate.size() < deathFrame) candidate.resize(deathFrame, {false});
        int first = std::max(0, deathFrame - mutationWindow);
        int last = std::min(deathFrame - 1, (int)candidate.size() - 1);
        int span = last - first + 1;
        if (span <= 0) return candidate;

        auto nextRandom = [&seed]() {
            seed = seed * 747796405u + 2891336453u;
            uint32_t value = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737u;
            return (value >> 22) ^ value;
        };
        int flips = 1 + (int)(nextRandom() % (uint32_t)std::min(4, span));
        std::vector<bool> changed((size_t)span, false);
        for (int i = 0; i < flips; ++i) {
            int offset = (int)(nextRandom() % (uint32_t)span);
            while (changed[(size_t)offset]) offset = (offset + 1) % span;
            changed[(size_t)offset] = true;
            candidate[(size_t)(first + offset)].hold = !candidate[(size_t)(first + offset)].hold;
        }
        return candidate;
    }

    void updateScoreLabel() {
        if (!s_scoreLabel) return;
        s_scoreLabel->setString(("LEARNING SCORE: " + std::to_string(s_score)).c_str());
    }

    void drawStoredTrajectory() {
        if (!s_trajectoryNode) return;

        s_trajectoryNode->clear();
        auto drawPath = [](std::vector<TrailPoint> const& path, ccColor4F lineColor, ccColor4F clickColor) {
            if (path.empty()) return;
            for (size_t i = 1; i < path.size(); ++i) {
                s_trajectoryNode->drawSegment(path[i - 1].position, path[i].position, 1.5f, lineColor);
            }
            for (auto const& point : path) {
                if (point.click) s_trajectoryNode->drawDot(point.position, 4.0f, clickColor);
            }
        };

        drawPath(s_bestTrajectory, ccc4f(1.0f, 0.72f, 0.12f, 0.85f), ccc4f(1.0f, 0.92f, 0.25f, 1.0f));
        drawPath(s_segment.bestTrajectory, ccc4f(1.0f, 0.72f, 0.12f, 0.85f), ccc4f(1.0f, 0.92f, 0.25f, 1.0f));
        drawPath(s_currentTrajectory, ccc4f(0.15f, 0.8f, 1.0f, 0.8f), ccc4f(1.0f, 0.25f, 0.25f, 1.0f));
    }

    void beginTrajectoryAttempt() {
        s_currentTrajectory.clear();
        s_hasLastTrajectoryPoint = false;
        s_lastHold = false;
        drawStoredTrajectory();
    }

    void recordTrajectoryPoint(CCPoint position, bool click) {
        if (s_trajectoryNode && s_hasLastTrajectoryPoint) {
            s_trajectoryNode->drawSegment(
                s_lastTrajectoryPoint,
                position,
                1.5f,
                ccc4f(0.15f, 0.8f, 1.0f, 0.8f)
            );
        }
        if (s_trajectoryNode && click) {
            s_trajectoryNode->drawDot(position, 4.0f, ccc4f(1.0f, 0.25f, 0.25f, 1.0f));
        }
        s_currentTrajectory.push_back({position, click});
        s_lastTrajectoryPoint = position;
        s_hasLastTrajectoryPoint = true;
    }

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
        BotState::s_attemptsCount = 0;
        BotState::s_lastDeathPercent = -1000.0f;
        BotState::s_sameSpotDeaths = 0;
        BotState::s_score = 0;
        BotState::s_bestDistanceX = 0.0f;
        BotState::s_bestPath.clear();
        BotState::s_candidatePath.clear();
        BotState::s_bestTrajectory.clear();
        BotState::s_segment = {};
        BotState::s_segmentAttempt.clear();
        BotState::s_segmentCandidate.clear();
        BotState::s_trainingSegment = false;
        BotState::s_startPosTraining = this->m_startPosObject != nullptr;
        BotState::s_autoPracticeTraining = false;
        BotState::s_autoCheckpointPlaced = false;
        BotState::s_noProgressDeaths = 0;
        BotState::s_segmentPlaybackFrame = 0;
        BotState::s_segmentPlaybackActive = false;
        BotState::s_segmentPlaybackDone = false;
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

        auto legend = CCLabelBMFont::create("TRAIL: red=click  gold=best  cyan=current", "chatFont.fnt");
        legend->setScale(0.34f);
        legend->setAnchorPoint(ccp(0.0f, 0.5f));
        legend->setPosition(ccp(12.0f, -15.0f));
        hudNode->addChild(legend);
        BotState::s_trajectoryLegend = legend;

        auto scoreLabel = CCLabelBMFont::create("LEARNING SCORE: 0", "chatFont.fnt");
        scoreLabel->setScale(0.34f);
        scoreLabel->setAnchorPoint(ccp(0.0f, 0.5f));
        scoreLabel->setPosition(ccp(12.0f, -27.0f));
        hudNode->addChild(scoreLabel);
        BotState::s_scoreLabel = scoreLabel;

        auto trajectory = CCDrawNode::create();
        trajectory->setID("ai-bot-trajectory"_spr);
        trajectory->setVisible(Mod::get()->getSettingValue<bool>("show-trajectory"));
        BotState::s_trajectoryNode = trajectory;
        if (this->m_objectLayer) this->m_objectLayer->addChild(trajectory, 1000);

        auto hitbox = CCDrawNode::create();
        hitbox->setID("ai-bot-hitbox"_spr);
        hitbox->setVisible(Mod::get()->getSettingValue<bool>("show-hitbox"));
        BotState::s_hitboxNode = hitbox;
        if (this->m_objectLayer) this->m_objectLayer->addChild(hitbox, 1001);
        BotState::beginTrajectoryAttempt();

        this->addChild(hudNode);
        BotState::refreshIndicator();

        return true;
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        BotState::s_currentFrame = 0;
        BotState::s_lastHold = false;
        BotState::s_cpsCooldown = 0.0f;
        BotState::s_segmentPlaybackFrame = 0;
        BotState::s_segmentPlaybackActive = false;
        BotState::s_segmentPlaybackDone = false;

        bool checkpointRespawn = this->m_isPracticeMode && this->m_checkpointArray && this->m_checkpointArray->count() > 0;
        if (BotState::s_autoPracticeTraining && BotState::s_autoCheckpointPlaced) checkpointRespawn = true;
        BotState::s_trainingSegment = BotState::s_startPosTraining || checkpointRespawn;
        if (BotState::s_trainingSegment) {
            float startPercent = this->getCurrentPercent();
            if (std::abs(BotState::s_segment.startPercent - startPercent) > 1.0f) {
                BotState::s_segment = {};
            }
            BotState::s_segment.startPercent = startPercent;
            BotState::s_segment.bestPercent = std::max(BotState::s_segment.bestPercent, startPercent);
            BotState::s_segmentAttempt = BotState::s_segmentCandidate.empty()
                ? BotState::s_segment.bestInputs : BotState::s_segmentCandidate;
        } else {
            BotState::s_currentAttempt = BotState::s_candidatePath.empty()
                ? BotState::s_bestPath : BotState::s_candidatePath;
        }
        this->handleButton(false, 1, true);
        BotState::beginTrajectoryAttempt();
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);

        // Keep the HUD in sync when the bot is toggled from the pause menu.
        BotState::refreshIndicator();

        if (!BotState::s_enabled) return;
        if (!this->m_player1 || this->m_player1->m_isDead) return;

        float progress = this->getCurrentPercent();
        if (BotState::s_startPosTraining && !BotState::s_trainingSegment) {
            BotState::s_trainingSegment = true;
            BotState::s_segment.startPercent = progress;
            BotState::s_segment.bestPercent = progress;
            BotState::s_segmentAttempt = BotState::s_segmentCandidate.empty()
                ? BotState::s_segment.bestInputs : BotState::s_segmentCandidate;
            BotState::s_currentFrame = 0;
        }

        if (BotState::s_autoPracticeTraining && !BotState::s_autoCheckpointPlaced && progress >= BotState::s_checkpointTargetPercent) {
            this->createCheckpoint();
            BotState::s_autoCheckpointPlaced = true;
            BotState::s_trainingSegment = true;
            if (std::abs(BotState::s_segment.startPercent - progress) > 1.0f) {
                BotState::s_segment = {};
            }
            BotState::s_segment.startPercent = progress;
            BotState::s_segment.bestPercent = progress;
            BotState::s_segmentAttempt = BotState::s_segmentCandidate.empty()
                ? BotState::s_segment.bestInputs : BotState::s_segmentCandidate;
            BotState::s_currentFrame = 0;
            BotState::beginTrajectoryAttempt();
        }

        // Keep frame zero aligned with the first input and grow before indexing.
        auto frame = BotState::s_currentFrame;
        auto& inputPath = BotState::s_trainingSegment ? BotState::s_segmentAttempt : BotState::s_currentAttempt;
        if (frame >= (int)inputPath.size()) {
            auto knownLength = inputPath.size();
            inputPath.resize(frame + 1, {false});
            // The old path ends at the best run's frontier. Explore beyond it
            // with a varied, sparse tap pattern so the bot can discover later
            // parts of the level instead of coasting forever without input.
            auto seed = (uint32_t)(frame * 747796405u + (BotState::s_attemptsCount + 1) * 2891336453u);
            seed = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737u;
            seed = (seed >> 22) ^ seed;
            inputPath[frame].hold = frame >= (int)knownLength && (seed % 100u) < 18u;
        }

        bool hold = inputPath[frame].hold;
        if (!BotState::s_trainingSegment && !BotState::s_segment.bestInputs.empty()) {
            if (!BotState::s_segmentPlaybackActive && !BotState::s_segmentPlaybackDone && progress >= BotState::s_segment.startPercent) {
                BotState::s_segmentPlaybackActive = true;
                BotState::s_segmentPlaybackFrame = 0;
            }
            if (BotState::s_segmentPlaybackActive) {
                if (BotState::s_segmentPlaybackFrame < (int)BotState::s_segment.bestInputs.size()) {
                    hold = BotState::s_segment.bestInputs[BotState::s_segmentPlaybackFrame++].hold;
                } else {
                    BotState::s_segmentPlaybackActive = false;
                    BotState::s_segmentPlaybackDone = true;
                }
            }
        }

        // Send only state changes: consecutive true frames hold, alternating
        // frames produce rapid taps without repeated press events.
        int cpsLimit = (int)Mod::get()->getSettingValue<int64_t>("cps-limit");
        cpsLimit = std::max(1, cpsLimit);
        BotState::s_cpsCooldown = std::max(0.0f, BotState::s_cpsCooldown - dt);
        if (hold && !BotState::s_lastHold && BotState::s_cpsCooldown > 0.0f) {
            hold = false;
        }
        bool click = hold && !BotState::s_lastHold;
        if (click) BotState::s_cpsCooldown = 1.0f / (float)cpsLimit;
        if (hold != BotState::s_lastHold) {
            this->handleButton(hold, 1, true);
        }
        BotState::s_lastHold = hold;

        auto playerRect = m_player1->getObjectRect();
        auto position = ccp(playerRect.origin.x + playerRect.size.width * 0.5f,
                            playerRect.origin.y + playerRect.size.height * 0.5f);
        BotState::recordTrajectoryPoint(position, click);

        if (BotState::s_hitboxNode) {
            BotState::s_hitboxNode->clear();
            if (BotState::s_hitboxNode->isVisible()) {
                auto left = playerRect.origin.x;
                auto bottom = playerRect.origin.y;
                auto right = left + playerRect.size.width;
                auto top = bottom + playerRect.size.height;
                auto color = ccc4f(0.25f, 1.0f, 0.25f, 0.9f);
                BotState::s_hitboxNode->drawSegment(ccp(left, bottom), ccp(right, bottom), 1.5f, color);
                BotState::s_hitboxNode->drawSegment(ccp(right, bottom), ccp(right, top), 1.5f, color);
                BotState::s_hitboxNode->drawSegment(ccp(right, top), ccp(left, top), 1.5f, color);
                BotState::s_hitboxNode->drawSegment(ccp(left, top), ccp(left, bottom), 1.5f, color);
            }
        }
        if (BotState::s_trainingSegment) BotState::s_segmentAttempt[frame].hold = hold;
        else BotState::s_currentAttempt[frame].hold = hold;
        BotState::s_currentFrame++;
    }

    void onExit() {
        // These nodes belong to this PlayLayer. Clear the cached pointers
        // before the layer releases its children so setting listeners cannot
        // touch freed UI objects after leaving a level.
        BotState::s_indicatorDot = nullptr;
        BotState::s_indicatorLabel = nullptr;
        BotState::s_trajectoryLegend = nullptr;
        BotState::s_scoreLabel = nullptr;
        BotState::s_trajectoryNode = nullptr;
        BotState::s_hitboxNode = nullptr;
        BotState::s_hasLastTrajectoryPoint = false;
        PlayLayer::onExit();
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
        auto pl = PlayLayer::get();
        float currentPercent = pl ? pl->getCurrentPercent() : 0.0f;
        int deathFrame = BotState::s_currentFrame;
        int mutationWindow = (int)Mod::get()->getSettingValue<int64_t>("learning-rate");
        float previousBest = BotState::s_trainingSegment
            ? BotState::s_segment.bestPercent : BotState::s_bestDistanceX;
        bool improved = currentPercent > previousBest;
        if (improved) {
            auto gain = currentPercent - previousBest;
            BotState::s_score += std::max(1, (int)std::round(gain * 10.0f));
        } else {
            BotState::s_score = std::max(0, BotState::s_score - 1);
        }
        BotState::updateScoreLabel();

        if (BotState::s_trainingSegment) {
            if (improved) {
                BotState::s_segment.bestPercent = currentPercent;
                BotState::s_segment.bestInputs = BotState::s_segmentAttempt;
                BotState::s_segment.bestTrajectory = BotState::s_currentTrajectory;
                BotState::drawStoredTrajectory();
            }
            auto const& champion = BotState::s_segment.bestInputs.empty()
                ? BotState::s_segmentAttempt : BotState::s_segment.bestInputs;
            BotState::s_segmentCandidate = BotState::makeCandidate(
                champion, deathFrame, mutationWindow,
                (uint32_t)(BotState::s_attemptsCount * 314159u + deathFrame)
            );

            // Once the trained section beats the previous full-run record,
            // leave Practice Mode and test the saved inputs in a full run.
            bool segmentMastered = BotState::s_autoPracticeTraining &&
                BotState::s_segment.bestPercent >= BotState::s_bestDistanceX + 2.0f;
            if (segmentMastered && pl) {
                pl->togglePracticeMode(false);
                BotState::s_autoPracticeTraining = false;
                BotState::s_autoCheckpointPlaced = false;
                BotState::s_trainingSegment = false;
                BotState::s_segmentPlaybackActive = false;
                if (Mod::get()->getSettingValue<bool>("auto-reset")) {
                    pl->resetLevelFromStart();
                }
                return;
            }
        }
        else {
            if (improved) {
                BotState::s_bestDistanceX = currentPercent;
                BotState::s_bestPath = BotState::s_currentAttempt;
                BotState::s_bestTrajectory = BotState::s_currentTrajectory;
                BotState::s_noProgressDeaths = 0;
                BotState::drawStoredTrajectory();
            } else {
                BotState::s_noProgressDeaths++;
            }
            auto const& champion = BotState::s_bestPath.empty()
                ? BotState::s_currentAttempt : BotState::s_bestPath;
            BotState::s_candidatePath = BotState::makeCandidate(
                champion, deathFrame, mutationWindow,
                (uint32_t)(BotState::s_attemptsCount * 314159u + deathFrame)
            );

            // After five deaths without a new record, add a Practice
            // checkpoint just before the stuck section and train from there.
            if (pl && !pl->m_isPracticeMode && BotState::s_noProgressDeaths >= 5) {
                pl->togglePracticeMode(true);
                BotState::s_autoPracticeTraining = true;
                BotState::s_autoCheckpointPlaced = false;
                BotState::s_checkpointTargetPercent = std::max(0.0f, BotState::s_bestDistanceX - 2.0f);
                BotState::s_noProgressDeaths = 0;
            }
        }

        if (pl && pl->m_isPracticeMode && pl->m_checkpointArray && pl->m_checkpointArray->count() > 0 &&
            Mod::get()->getSettingValue<bool>("remove-stuck-checkpoints")) {
            if (std::abs(currentPercent - BotState::s_lastDeathPercent) <= 0.25f) {
                BotState::s_sameSpotDeaths++;
            } else {
                BotState::s_sameSpotDeaths = 1;
            }
            BotState::s_lastDeathPercent = currentPercent;

            if (BotState::s_sameSpotDeaths >= 20) {
                // In Geode's binding, false removes the newest checkpoint.
                pl->removeCheckpoint(false);
                BotState::s_sameSpotDeaths = 0;
                BotState::s_lastDeathPercent = -1000.0f;
                if (BotState::s_autoPracticeTraining) {
                    BotState::s_autoPracticeTraining = false;
                    BotState::s_autoCheckpointPlaced = false;
                    BotState::s_trainingSegment = false;
                }
            }
        } else {
            BotState::s_sameSpotDeaths = 0;
            BotState::s_lastDeathPercent = -1000.0f;
        }

        // Jesli wlaczony szybki restart
        if (Mod::get()->getSettingValue<bool>("auto-reset")) {
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

    void onToggleAI(CCObject* sender) {
        BotState::toggle();

        // Update the button in place. Rebuilding the whole PauseLayer from
        // inside its own button callback destroys the active menu item and
        // can crash the game.
        auto btn = static_cast<CCMenuItemSpriteExtra*>(sender);
        auto spr = CircleButtonSprite::create(
            CCLabelBMFont::create("AI", "bigFont.fnt"),
            BotState::s_enabled ? CircleBaseColor::Green : CircleBaseColor::Gray,
            CircleBaseSize::Small
        );
        btn->setNormalImage(spr);

        Notification::create(
            BotState::s_enabled ? "AI Bot: WLACZONY (Cheat Indicator: CZERWONY)" : "AI Bot: WYLACZONY (Cheat Indicator: ZIELONY)",
            BotState::s_enabled ? NotificationIcon::Warning : NotificationIcon::Success
        )->show();

    }
};

// -------------------------------------------------------------
// 4. Rejestracja nasluchiwania zmian w menu modow Geode
// -------------------------------------------------------------
$on_mod(Loaded) {
    listenForSettingChanges<bool>("bot-enabled", [](bool val) {
        BotState::s_enabled = val;
        BotState::refreshIndicator();
    });
    listenForSettingChanges<bool>("cheat-indicator", [](bool val) {
        BotState::refreshIndicator();
    });
    listenForSettingChanges<bool>("show-hitbox", [](bool visible) {
        if (BotState::s_hitboxNode) BotState::s_hitboxNode->setVisible(visible);
    });
    listenForSettingChanges<bool>("show-trajectory", [](bool visible) {
        if (!BotState::s_trajectoryNode) return;
        BotState::s_trajectoryNode->setVisible(visible);
        if (visible) BotState::drawStoredTrajectory();
    });
}

