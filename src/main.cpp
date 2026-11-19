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
    float percent = 0.0f;
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
    inline double s_normalScore = 0.0;
    inline double s_practiceScore = 0.0;
    inline constexpr double s_scoreLimit = 2147483647.0;
    inline CCDrawNode* s_trajectoryNode = nullptr;
    inline CCDrawNode* s_hitboxNode = nullptr;
    inline CCDrawNode* s_predictionNode = nullptr;
    inline std::vector<TrailPoint> s_currentTrajectory;
    inline std::vector<TrailPoint> s_bestTrajectory;
    inline bool s_hasLastTrajectoryPoint = false;
    inline bool s_lastHold = false;
    inline float s_cpsCooldown = 0.0f;
    inline CCPoint s_lastTrajectoryPoint = CCPointZero;
    inline uint32_t s_exploreRandomSeed = 1;
    inline int s_explorationRunCount = 0;
    inline bool s_exploreHold = false;
    inline float s_exploreRemainingSeconds = 0.0f;
    inline LearnedSegment s_segment;
    inline std::vector<FrameDecision> s_segmentCandidate;
    inline std::vector<FrameDecision> s_segmentAttempt;
    inline bool s_trainingSegment = false;
    inline bool s_startPosTraining = false;
    inline bool s_autoPracticeTraining = false;
    inline int s_noProgressDeaths = 0;
    inline int s_segmentPlaybackFrame = 0;
    inline bool s_segmentPlaybackActive = false;
    inline bool s_segmentPlaybackDone = false;

    // Elementy wizualne Cheat Indicatora
    inline CCDrawNode* s_indicatorDot = nullptr;
    inline CCLabelBMFont* s_indicatorLabel = nullptr;
    inline CCLabelBMFont* s_trajectoryLegend = nullptr;
    inline CCLabelBMFont* s_scoreLabel = nullptr;

    int inputMemoryLimit() {
        int limit = (int)Mod::get()->getSettingValue<int64_t>("input-memory-ticks");
        limit = std::clamp(limit, 30000, 250000);
        int budgetMb = (int)Mod::get()->getSettingValue<int64_t>("memory-budget-mb");
        budgetMb = std::clamp(budgetMb, 1, 50);
        // Reserve for the best/current/candidate input and trajectory vectors,
        // including their temporary copies during a new best run.
        constexpr size_t estimatedBytesPerTick =
            8 * sizeof(FrameDecision) + 4 * sizeof(TrailPoint);
        auto budgetTicks = ((size_t)budgetMb * 1024u * 1024u) / estimatedBytesPerTick;
        return std::max(1, std::min(limit, (int)budgetTicks));
    }

    std::vector<FrameDecision> makeCandidate(
        std::vector<FrameDecision> const& champion,
        int deathFrame,
        int mutationWindow,
        uint32_t seed
    ) {
        auto candidate = champion;
        int memoryLimit = inputMemoryLimit();
        if ((int)candidate.size() > memoryLimit) candidate.resize(memoryLimit);
        deathFrame = std::clamp(deathFrame, 0, memoryLimit);
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
        int flips = 1 + (int)(nextRandom() % (uint32_t)std::min(2, span));
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
        auto formatScore = [](double score) {
            auto text = std::to_string(std::round(score * 100.0) / 100.0);
            while (text.find('.') != std::string::npos && text.back() == '0') text.pop_back();
            if (!text.empty() && text.back() == '.') text.pop_back();
            return text;
        };
        auto total = s_normalScore * 2.0 + s_practiceScore;
        auto text = "SCORE NORMAL x2: " + formatScore(s_normalScore) +
            " | PRACTICE: " + formatScore(s_practiceScore) +
            " | TOTAL: " + formatScore(total);
        s_scoreLabel->setString(text.c_str());
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
        if (!s_bestTrajectory.empty()) {
            s_trajectoryNode->drawDot(s_bestTrajectory.back().position, 6.0f,
                                      ccc4f(1.0f, 1.0f, 1.0f, 1.0f));
        }
    }

    void drawPlayerHitbox(CCRect const& playerRect) {
        if (!s_hitboxNode) return;
        s_hitboxNode->clear();
        if (!s_hitboxNode->isVisible()) return;

        auto left = playerRect.origin.x;
        auto bottom = playerRect.origin.y;
        auto right = left + playerRect.size.width;
        auto top = bottom + playerRect.size.height;
        auto color = ccc4f(0.25f, 1.0f, 0.25f, 0.9f);
        s_hitboxNode->drawSegment(ccp(left, bottom), ccp(right, bottom), 1.5f, color);
        s_hitboxNode->drawSegment(ccp(right, bottom), ccp(right, top), 1.5f, color);
        s_hitboxNode->drawSegment(ccp(right, top), ccp(left, top), 1.5f, color);
        s_hitboxNode->drawSegment(ccp(left, top), ccp(left, bottom), 1.5f, color);
    }

    void drawPredictedTrajectory() {
        if (!s_predictionNode) return;
        s_predictionNode->clear();
        if (!s_predictionNode->isVisible() || s_currentTrajectory.size() < 3) return;

        auto count = s_currentTrajectory.size();
        auto p0 = s_currentTrajectory[count - 3].position;
        auto p1 = s_currentTrajectory[count - 2].position;
        auto p2 = s_currentTrajectory[count - 1].position;
        constexpr float tickRate = 60.0f;
        int predictionSteps = (int)Mod::get()->getSettingValue<int64_t>("prediction-steps");
        predictionSteps = std::clamp(predictionSteps, 5, 50);
        auto velocity = ccp((p2.x - p1.x) * tickRate, (p2.y - p1.y) * tickRate);
        auto acceleration = ccp(
            (p2.x - 2.0f * p1.x + p0.x) * tickRate * tickRate,
            (p2.y - 2.0f * p1.y + p0.y) * tickRate * tickRate
        );

        // Clamp spikes from portals/teleports so they do not send the preview
        // across the whole screen for one frame.
        auto accelerationLength = std::sqrt(acceleration.x * acceleration.x + acceleration.y * acceleration.y);
        if (accelerationLength > 5000.0f) {
            auto scale = 5000.0f / accelerationLength;
            acceleration = ccp(acceleration.x * scale, acceleration.y * scale);
        }

        auto previous = p2;
        auto color = ccc4f(0.85f, 0.2f, 1.0f, 0.9f);

        // When this level has a remembered route, use its future positions as
        // the primary forecast. Align the stored path to the icon's current
        // position so the overlay follows the live run rather than the old run.
        auto const& memory = s_trainingSegment && !s_segment.bestTrajectory.empty()
            ? s_segment.bestTrajectory : s_bestTrajectory;
        if (!memory.empty()) {
            auto currentPercent = s_currentTrajectory.back().percent;
            auto it = std::lower_bound(memory.begin(), memory.end(), currentPercent,
                [](TrailPoint const& point, float target) { return point.percent < target; });
            if (it != memory.end()) {
                auto memoryStart = it->position;
                auto offset = ccp(p2.x - memoryStart.x, p2.y - memoryStart.y);
                auto previousMemory = p2;
                int drawn = 0;
                for (++it; it != memory.end() && drawn < predictionSteps; ++it, ++drawn) {
                    auto nextMemory = ccp(it->position.x + offset.x, it->position.y + offset.y);
                    s_predictionNode->drawSegment(previousMemory, nextMemory, 2.0f, color);
                    previousMemory = nextMemory;
                }
                if (drawn > 0) {
                    s_predictionNode->drawDot(previousMemory, 3.5f, color);
                    return;
                }
            }
        }

        for (int step = 1; step <= predictionSteps; ++step) {
            auto time = (float)step / tickRate;
            auto next = ccp(
                p2.x + velocity.x * time + 0.5f * acceleration.x * time * time,
                p2.y + velocity.y * time + 0.5f * acceleration.y * time * time
            );
            s_predictionNode->drawSegment(previous, next, 2.0f, color);
            previous = next;
        }
        s_predictionNode->drawDot(previous, 3.5f, color);
    }

    uint32_t nextExploreRandom() {
        s_exploreRandomSeed = s_exploreRandomSeed * 747796405u + 2891336453u;
        uint32_t value = ((s_exploreRandomSeed >> ((s_exploreRandomSeed >> 28) + 4)) ^ s_exploreRandomSeed) * 277803737u;
        return (value >> 22) ^ value;
    }

    void resetExplorationPattern() {
        ++s_explorationRunCount;
        s_exploreRandomSeed = (uint32_t)(s_attemptsCount * 314159u + s_explorationRunCount * 2718281u + 1u);
        s_exploreHold = false;
        s_exploreRemainingSeconds = 0.0f;
    }

    bool nextExploreHold(float dt) {
        s_exploreRemainingSeconds -= std::max(0.0f, dt);
        if (s_exploreRemainingSeconds <= 0.0f) {
            auto choice = nextExploreRandom() % 100u;
            s_exploreHold = choice < 70u;
            if (s_exploreHold) {
                s_exploreRemainingSeconds = 0.1f + (float)(nextExploreRandom() % 1901u) / 1000.0f;
            } else if (choice < 90u) {
                // Sometimes leave the input released long enough to observe
                // whether gravity and momentum carry the player forward.
                s_exploreRemainingSeconds = 0.5f + (float)(nextExploreRandom() % 1501u) / 1000.0f;
            } else {
                s_exploreRemainingSeconds = 0.1f + (float)(nextExploreRandom() % 401u) / 1000.0f;
            }
        }
        return s_exploreHold;
    }

    size_t bestFrameAtPercent(float percent) {
        if (s_bestTrajectory.empty()) return 0;
        auto it = std::lower_bound(
            s_bestTrajectory.begin(), s_bestTrajectory.end(), percent,
            [](TrailPoint const& point, float target) { return point.percent < target; }
        );
        return (size_t)std::distance(s_bestTrajectory.begin(), it);
    }

    bool guessFromSimilarMemory(
        CCPoint position,
        std::vector<FrameDecision> const& inputs,
        std::vector<TrailPoint> const& trajectory,
        bool& guessedHold
    ) {
        auto count = std::min(inputs.size(), trajectory.size());
        if (count < 2) return false;

        float closestDistanceSquared = 96.0f * 96.0f;
        size_t closest = count;
        for (size_t i = 0; i < count; ++i) {
            auto dx = trajectory[i].position.x - position.x;
            auto dy = trajectory[i].position.y - position.y;
            auto distanceSquared = dx * dx + dy * dy;
            if (distanceSquared < closestDistanceSquared) {
                closestDistanceSquared = distanceSquared;
                closest = i;
            }
        }
        if (closest == count || closest + 1 >= count) return false;

        // Reuse the next action from a remembered point in a similar place.
        // This gives exploration a data-based guess before it falls back to
        // random holds and releases.
        guessedHold = inputs[closest + 1].hold;
        return true;
    }

    void seedSegmentFromBest(float startPercent) {
        if (!s_segment.bestInputs.empty() || s_bestPath.empty()) return;

        auto startFrame = bestFrameAtPercent(startPercent);
        if (startFrame >= s_bestPath.size()) return;
        s_segment.bestInputs.assign(s_bestPath.begin() + startFrame, s_bestPath.end());
        if (startFrame < s_bestTrajectory.size()) {
            s_segment.bestTrajectory.assign(s_bestTrajectory.begin() + startFrame, s_bestTrajectory.end());
        }
    }

    void beginTrajectoryAttempt() {
        s_currentTrajectory.clear();
        s_hasLastTrajectoryPoint = false;
        s_lastHold = false;
        resetExplorationPattern();
        drawStoredTrajectory();
        drawPredictedTrajectory();
    }

    void recordTrajectoryPoint(CCPoint position, bool click, float percent) {
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
        auto maxSamples = (size_t)inputMemoryLimit();
        if (s_currentTrajectory.size() >= maxSamples && maxSamples > 1) {
            // Keep the newest live observations when the configured memory
            // window fills; older current-run samples are the first to go.
            auto discard = std::max<size_t>(1, maxSamples / 16);
            s_currentTrajectory.erase(s_currentTrajectory.begin(),
                s_currentTrajectory.begin() + std::min(discard, s_currentTrajectory.size()));
        }
        s_currentTrajectory.push_back({position, click, percent});
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
        BotState::s_normalScore = 0;
        BotState::s_practiceScore = 0;
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

        auto legend = CCLabelBMFont::create("TRAIL: red=click gold=best white=memory-end cyan=current purple=prediction", "chatFont.fnt");
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
        BotState::updateScoreLabel();

        auto trajectory = CCDrawNode::create();
        trajectory->setID("ai-bot-trajectory"_spr);
        trajectory->setVisible(Mod::get()->getSettingValue<bool>("show-trajectory"));
        BotState::s_trajectoryNode = trajectory;
        if (this->m_objectLayer) this->m_objectLayer->addChild(trajectory, 1000);

        auto prediction = CCDrawNode::create();
        prediction->setID("ai-bot-prediction"_spr);
        prediction->setVisible(Mod::get()->getSettingValue<bool>("show-prediction"));
        BotState::s_predictionNode = prediction;
        if (this->m_objectLayer) this->m_objectLayer->addChild(prediction, 1002);

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
        BotState::s_trainingSegment = BotState::s_startPosTraining || checkpointRespawn;
        if (BotState::s_trainingSegment) {
            float startPercent = this->getCurrentPercent();
            if (std::abs(BotState::s_segment.startPercent - startPercent) > 1.0f) {
                BotState::s_segment = {};
            }
            BotState::s_segment.startPercent = startPercent;
            BotState::s_segment.bestPercent = std::max(BotState::s_segment.bestPercent, startPercent);
            BotState::seedSegmentFromBest(startPercent);
            BotState::s_segmentAttempt = BotState::s_segmentCandidate.empty()
                ? BotState::s_segment.bestInputs : BotState::s_segmentCandidate;
        } else {
            BotState::s_currentAttempt = BotState::s_candidatePath.empty()
                ? BotState::s_bestPath : BotState::s_candidatePath;
        }
        this->handleButton(false, 1, true);
        if (this->m_player2) this->handleButton(false, 2, true);
        BotState::beginTrajectoryAttempt();
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);

        // Keep the HUD in sync when the bot is toggled from the pause menu.
        BotState::refreshIndicator();

        if (!this->m_player1 || this->m_player1->m_isDead) return;

        float progress = this->getCurrentPercent();
        auto playerRect = this->m_player1->getObjectRect();
        auto position = ccp(playerRect.origin.x + playerRect.size.width * 0.5f,
                            playerRect.origin.y + playerRect.size.height * 0.5f);
        if (!BotState::s_enabled) {
            // Visual tracking must continue when the AI is turned off from
            // the pause menu; only input control is disabled in this state.
            auto playerRect = this->m_player1->getObjectRect();
            auto position = ccp(playerRect.origin.x + playerRect.size.width * 0.5f,
                                playerRect.origin.y + playerRect.size.height * 0.5f);
            BotState::recordTrajectoryPoint(position, false, progress);
            BotState::drawPlayerHitbox(playerRect);
            BotState::drawPredictedTrajectory();
            return;
        }

        // Learning score is time based: every active AI tick earns 0.05 point.
        auto& learningScore = this->m_isPracticeMode
            ? BotState::s_practiceScore : BotState::s_normalScore;
        auto tickRewardHundredths = (int)Mod::get()->getSettingValue<int64_t>("learning-tick-reward-hundredths");
        tickRewardHundredths = std::clamp(tickRewardHundredths, 0, 100);
        learningScore = std::clamp(learningScore + tickRewardHundredths / 100.0,
                                   -BotState::s_scoreLimit, BotState::s_scoreLimit);
        BotState::updateScoreLabel();

        if (this->m_isPracticeMode &&
            Mod::get()->getSettingValue<bool>("exit-practice-at-percent")) {
            int exitPercent = (int)Mod::get()->getSettingValue<int64_t>("practice-exit-percent");
            exitPercent = std::clamp(exitPercent, 1, 100);
            float exitTarget = std::max((float)exitPercent, BotState::s_bestDistanceX + 0.01f);
            if (progress >= exitTarget) {
                // Keep the newly reached full-level pattern, then leave
                // Practice Mode and restart from zero for the next attempt.
                BotState::s_bestDistanceX = progress;
                BotState::s_bestPath = BotState::s_currentAttempt;
                BotState::s_candidatePath = BotState::s_bestPath;
                BotState::s_bestTrajectory = BotState::s_currentTrajectory;
                BotState::drawStoredTrajectory();
                this->togglePracticeMode(false);
                BotState::s_autoPracticeTraining = false;
                BotState::s_trainingSegment = false;
                BotState::s_segmentPlaybackActive = false;
                BotState::s_segmentPlaybackDone = false;
                this->resetLevelFromStart();
                return;
            }
        }
        if (BotState::s_startPosTraining && !BotState::s_trainingSegment) {
            BotState::s_trainingSegment = true;
            BotState::s_segment.startPercent = progress;
            BotState::s_segment.bestPercent = progress;
            BotState::s_segmentAttempt = BotState::s_segmentCandidate.empty()
                ? BotState::s_segment.bestInputs : BotState::s_segmentCandidate;
            BotState::s_currentFrame = 0;
        }

        // Keep frame zero aligned with the first input and grow before indexing.
        auto frame = BotState::s_currentFrame;
        auto& inputPath = BotState::s_trainingSegment ? BotState::s_segmentAttempt : BotState::s_currentAttempt;
        int memoryLimit = BotState::inputMemoryLimit();
        if ((int)inputPath.size() > memoryLimit) inputPath.resize(memoryLimit);
        auto knownLength = inputPath.size();
        bool canStoreInput = frame < memoryLimit;
        if (canStoreInput && frame >= (int)inputPath.size()) {
            inputPath.resize(frame + 1, {false});
        }

        bool hold = false;
        if (frame < (int)knownLength) {
            hold = inputPath[frame].hold;
        } else {
            // The old path ends at the best run's frontier. Explore beyond it
            // by first looking for a similar remembered position and reusing
            // its next click decision. Use a varied pattern when memory has no
            // nearby match. Decisions are retained up to the tick limit.
            hold = BotState::nextExploreHold(dt);
            auto const& guessInputs = BotState::s_trainingSegment && !BotState::s_segment.bestInputs.empty()
                ? BotState::s_segment.bestInputs : BotState::s_bestPath;
            auto const& guessTrajectory = BotState::s_trainingSegment && !BotState::s_segment.bestTrajectory.empty()
                ? BotState::s_segment.bestTrajectory : BotState::s_bestTrajectory;
            bool rememberedGuess = false;
            if (!BotState::guessFromSimilarMemory(position, guessInputs, guessTrajectory, rememberedGuess)) {
                rememberedGuess = hold;
            }
            hold = rememberedGuess;
            if (canStoreInput && frame < (int)inputPath.size()) inputPath[frame].hold = hold;
        }

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
        float cpsTarget = std::max(1.0f, (float)cpsLimit * 0.8f);
        BotState::s_cpsCooldown = std::max(0.0f, BotState::s_cpsCooldown - dt);
        if (hold && !BotState::s_lastHold && BotState::s_cpsCooldown > 0.0f) {
            hold = false;
        }
        bool click = hold && !BotState::s_lastHold;
        if (click) BotState::s_cpsCooldown = 1.0f / cpsTarget;
        if (hold != BotState::s_lastHold) {
            this->handleButton(hold, 1, true);
            if (this->m_player2) this->handleButton(hold, 2, true);
        }
        BotState::s_lastHold = hold;

        BotState::recordTrajectoryPoint(position, click, progress);

        BotState::drawPlayerHitbox(playerRect);
        BotState::drawPredictedTrajectory();
        if (frame < (int)inputPath.size()) inputPath[frame].hold = hold;
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
        BotState::s_predictionNode = nullptr;
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
        bool hasPracticeCheckpoint = pl && pl->m_isPracticeMode && pl->m_checkpointArray &&
            pl->m_checkpointArray->count() > 0;
        if (hasPracticeCheckpoint) {
            if (std::abs(currentPercent - BotState::s_lastDeathPercent) <= 0.25f) {
                BotState::s_sameSpotDeaths++;
            } else {
                BotState::s_sameSpotDeaths = 1;
            }
            BotState::s_lastDeathPercent = currentPercent;

            // After five deaths at the same Practice checkpoint, widen the
            // mutation window so the next attempt changes its click pattern.
            if (BotState::s_sameSpotDeaths >= 5) {
                mutationWindow = std::min(120, mutationWindow * 2);
            }
        } else {
            BotState::s_sameSpotDeaths = 0;
            BotState::s_lastDeathPercent = -1000.0f;
        }
        float previousBest = BotState::s_trainingSegment
            ? BotState::s_segment.bestPercent : BotState::s_bestDistanceX;
        bool improved = currentPercent > previousBest;
        bool practiceMode = pl && pl->m_isPracticeMode;
        double& score = practiceMode ? BotState::s_practiceScore : BotState::s_normalScore;
        auto deathPenalty = (int)Mod::get()->getSettingValue<int64_t>("learning-death-penalty");
        deathPenalty = std::clamp(deathPenalty, 0, 1000000);
        score = std::clamp(score - deathPenalty, -BotState::s_scoreLimit, BotState::s_scoreLimit);
        BotState::updateScoreLabel();
        int negativeScoreMutation = score < 0 ? std::min(120, (int)(-score / 2.0)) : 0;
        mutationWindow = std::min(120, mutationWindow + negativeScoreMutation);

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

            // After five deaths without a new record, enter Practice Mode
            // from zero and replay the remembered best click pattern.
            if (pl && !pl->m_isPracticeMode && BotState::s_noProgressDeaths >= 5) {
                pl->togglePracticeMode(true);
                BotState::s_autoPracticeTraining = true;
                BotState::s_noProgressDeaths = 0;
            }
        }

        if (pl && pl->m_isPracticeMode && pl->m_checkpointArray && pl->m_checkpointArray->count() > 0) {
            if (Mod::get()->getSettingValue<bool>("remove-stuck-checkpoints")) {
                int deathLimit = (int)Mod::get()->getSettingValue<int64_t>("same-spot-death-limit");
                deathLimit = std::max(1, deathLimit);
                if (BotState::s_sameSpotDeaths >= deathLimit) {
                    // In Geode's binding, false removes the newest checkpoint.
                    pl->removeCheckpoint(false);
                    BotState::s_sameSpotDeaths = 0;
                    BotState::s_lastDeathPercent = -1000.0f;
                    if (BotState::s_autoPracticeTraining) {
                        BotState::s_autoPracticeTraining = false;
                        BotState::s_trainingSegment = false;
                    }
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

        // postUpdate stops while the level is paused. Rebuild retained overlays
        // when the pause menu opens so the cyan trail and current hitbox remain visible.
        BotState::drawStoredTrajectory();
        if (auto playLayer = PlayLayer::get(); playLayer && playLayer->m_player1) {
            BotState::drawPlayerHitbox(playLayer->m_player1->getObjectRect());
            BotState::drawPredictedTrajectory();
        }

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
    listenForSettingChanges<bool>("show-prediction", [](bool visible) {
        if (BotState::s_predictionNode) {
            BotState::s_predictionNode->setVisible(visible);
            if (visible) BotState::drawPredictedTrajectory();
        }
    });
}

