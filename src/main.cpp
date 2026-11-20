#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/modify/PauseLayer.hpp>
#include <Geode/binding/StartPosObject.hpp>
#include <Geode/ui/Notification.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <array>
#include <string>

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

constexpr size_t NEURAL_INPUTS = 7;
struct NeuralSample {
    std::array<float, NEURAL_INPUTS> features{};
    bool hold = false;
    bool exploratory = false;
};

namespace BotState {
    inline bool s_enabled = false;
    inline bool s_practiceMode = false;
    inline int s_currentFrame = 0;
    inline std::vector<FrameDecision> s_bestPath;
    inline std::vector<FrameDecision> s_candidatePath;
    inline std::vector<FrameDecision> s_currentAttempt;
    inline float s_bestDistanceX = 0.0f;
    inline std::vector<FrameDecision> s_bestPracticePath;
    inline std::vector<FrameDecision> s_candidatePracticePath;
    inline std::vector<TrailPoint> s_bestPracticeTrajectory;
    inline float s_bestPracticeDistanceX = 0.0f;
    inline int s_attemptsCount = 0;
    inline float s_lastDeathPercent = -1000.0f;
    inline int s_sameSpotDeaths = 0;
    inline double s_normalScore = 0.0;
    inline double s_practiceScore = 0.0;
    inline constexpr double s_scoreLimit = 2147483647.0;
    inline CCDrawNode* s_trajectoryNode = nullptr;
    inline CCDrawNode* s_hitboxNode = nullptr;
    inline CCDrawNode* s_hazardNode = nullptr;
    inline CCDrawNode* s_predictionNode = nullptr;
    inline std::vector<CCRect> s_hazardRects;
    inline std::vector<TrailPoint> s_currentTrajectory;
    inline std::vector<TrailPoint> s_bestTrajectory;
    inline bool s_hasLastTrajectoryPoint = false;
    inline bool s_lastHold = false;
    inline float s_cpsCooldown = 0.0f;
    inline int s_naturalClickCooldownTicks = 0;
    inline float s_postDeathCooldown = 0.0f;
    inline CCPoint s_lastTrajectoryPoint = CCPointZero;
    inline uint32_t s_exploreRandomSeed = 1;
    inline int s_explorationRunCount = 0;
    inline bool s_exploreHold = false;
    inline float s_exploreRemainingSeconds = 0.0f;
    inline int s_startDelayTicks = 0;
    inline bool s_allowPracticeMemoryGuess = false;
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
    inline CCLabelBMFont* s_deathAnalysisLabel = nullptr;
    inline CCLabelBMFont* s_neuralMonitorLabel = nullptr;
    inline CCLabelBMFont* s_levelTimeLabel = nullptr;
    inline CCLabelBMFont* s_levelBestLabel = nullptr;
    inline float s_levelActiveSeconds = 0.0f;
    inline float s_levelTimerRefresh = 0.0f;
    inline float s_levelBestPercent = 0.0f;
    inline std::string s_levelBestKey;
    inline size_t s_activeNeurons = 0;
    inline size_t s_activeSynapses = 0;
    inline float s_meanActivation = 0.0f;
    inline std::vector<std::array<float, NEURAL_INPUTS>> s_neuralWeights;
    inline std::vector<float> s_neuralBiases;
    inline std::vector<float> s_neuralOutputWeights;
    inline std::vector<float> s_neuralActivationsScratch;
    inline std::vector<NeuralSample> s_neuralSamples;
    inline float s_neuralOutputBias = 0.0f;

    int inputMemoryLimit() {
        int limit = (int)Mod::get()->getSettingValue<int64_t>("input-memory-ticks");
        limit = std::clamp(limit, 30000, 250000);
        int budgetMb = (int)Mod::get()->getSettingValue<int64_t>("memory-budget-mb");
        budgetMb = std::clamp(budgetMb, 1, 50);
        // Reserve for the best/current/candidate input and trajectory vectors,
        // including their temporary copies during a new best run.
        constexpr size_t estimatedBytesPerTick =
            10 * sizeof(FrameDecision) + 5 * sizeof(TrailPoint) + sizeof(NeuralSample);
        auto budgetTicks = ((size_t)budgetMb * 1024u * 1024u) / estimatedBytesPerTick;
        return std::max(1, std::min(limit, (int)budgetTicks));
    }

    float sigmoid(float value) {
        value = std::clamp(value, -16.0f, 16.0f);
        return 1.0f / (1.0f + std::exp(-value));
    }

    uint32_t nextExploreRandom();

    void ensureNeuralNetwork() {
        int count = (int)Mod::get()->getSettingValue<int64_t>("neural-neurons");
        count = std::clamp(count, 64, 200000);
        if ((int)s_neuralWeights.size() == count) return;
        s_neuralWeights.assign(count, {});
        s_neuralBiases.assign(count, 0.0f);
        s_neuralOutputWeights.assign(count, 0.0f);
        s_neuralActivationsScratch.resize(count);
        for (int neuron = 0; neuron < count; ++neuron) {
            for (auto& weight : s_neuralWeights[(size_t)neuron]) {
                weight = ((float)(nextExploreRandom() % 2001u) / 1000.0f - 1.0f) * 0.08f;
            }
            s_neuralBiases[(size_t)neuron] = ((float)(nextExploreRandom() % 2001u) / 1000.0f - 1.0f) * 0.02f;
            s_neuralOutputWeights[(size_t)neuron] = ((float)(nextExploreRandom() % 2001u) / 1000.0f - 1.0f) * 0.02f;
        }
        s_neuralOutputBias = 0.0f;
    }

    std::array<float, NEURAL_INPUTS> neuralFeatures(CCPoint position, float percent) {
        std::array<float, NEURAL_INPUTS> features{};
        features[0] = std::clamp(percent / 100.0f, 0.0f, 1.0f);
        features[1] = std::clamp(position.y / 600.0f, -1.0f, 1.0f);
        if (!s_currentTrajectory.empty()) {
            auto dy = position.y - s_currentTrajectory.back().position.y;
            features[2] = std::clamp(dy / 20.0f, -1.0f, 1.0f);
        }
        float nearestDistance = 10000.0f;
        float nearestDx = 1.0f;
        float nearestDy = 0.0f;
        bool foundHazard = false;
        for (auto const& rect : s_hazardRects) {
            auto dx = rect.origin.x - position.x;
            if (dx < -30.0f || dx > 300.0f) continue;
            if (dx < nearestDistance) {
                nearestDistance = dx;
                nearestDx = std::clamp(dx / 300.0f, -0.1f, 1.0f);
                nearestDy = std::clamp((rect.origin.y + rect.size.height * 0.5f - position.y) / 200.0f, -1.0f, 1.0f);
                foundHazard = true;
            }
        }
        features[3] = foundHazard ? nearestDx : 1.0f;
        features[4] = foundHazard ? nearestDy : 0.0f;
        features[5] = foundHazard ? 1.0f : 0.0f;
        features[6] = s_practiceMode ? 1.0f : 0.0f;
        return features;
    }

    float neuralPrediction(std::array<float, NEURAL_INPUTS> const& features,
                           std::vector<float>* activations = nullptr) {
        ensureNeuralNetwork();
        float output = s_neuralOutputBias;
        size_t activeNeurons = 0;
        size_t activeSynapses = 0;
        float activationTotal = 0.0f;
        size_t synapseCount = 0;
        if (activations) activations->clear();
        for (size_t i = 0; i < s_neuralWeights.size(); ++i) {
            float sum = s_neuralBiases[i];
            for (size_t j = 0; j < NEURAL_INPUTS; ++j) {
                auto weight = s_neuralWeights[i][j];
                sum += weight * features[j];
                ++synapseCount;
                if (std::abs(weight) >= 0.005f && std::abs(features[j]) >= 0.01f) ++activeSynapses;
            }
            // Clipped linear activation is much cheaper than tanh and bounded
            // activations keep the output layer numerically stable.
            auto activation = std::clamp(sum, -1.0f, 1.0f);
            activationTotal += std::abs(activation);
            if (std::abs(activation) >= 0.25f) ++activeNeurons;
            if (activations) activations->push_back(activation);
            auto outputWeight = s_neuralOutputWeights[i];
            ++synapseCount;
            if (std::abs(outputWeight) >= 0.005f && std::abs(activation) >= 0.01f) ++activeSynapses;
            output += outputWeight * activation;
        }
        if (!activations) {
            s_activeNeurons = activeNeurons;
            s_activeSynapses = activeSynapses;
            s_meanActivation = s_neuralWeights.empty() ? 0.0f : activationTotal / (float)s_neuralWeights.size();
            if (s_neuralMonitorLabel) {
                auto status = "SPIKES: " + std::to_string(s_activeNeurons) + " / " +
                    std::to_string(s_neuralWeights.size()) + "   SYNAPSES: " +
                    std::to_string(s_activeSynapses) + " / " + std::to_string(synapseCount) +
                    "   MEAN: " + std::to_string((int)std::round(s_meanActivation * 100.0f)) + "%";
                s_neuralMonitorLabel->setString(status.c_str());
            }
        }
        return sigmoid(output);
    }

    void trainNeuralSample(NeuralSample const& sample, float learningRate) {
        auto prediction = neuralPrediction(sample.features, &s_neuralActivationsScratch);
        float target = sample.hold ? 1.0f : 0.0f;
        float outputDelta = target - prediction;
        s_neuralOutputBias += learningRate * outputDelta;
        for (size_t i = 0; i < s_neuralActivationsScratch.size(); ++i) {
            auto activation = s_neuralActivationsScratch[i];
            auto derivative = std::abs(activation) < 1.0f ? 1.0f : 0.0f;
            auto hiddenDelta = outputDelta * s_neuralOutputWeights[i] * derivative;
            s_neuralOutputWeights[i] += learningRate * outputDelta * activation;
            for (size_t j = 0; j < NEURAL_INPUTS; ++j) {
                s_neuralWeights[i][j] += learningRate * hiddenDelta * sample.features[j];
            }
            s_neuralBiases[i] += learningRate * hiddenDelta;
        }
    }

    void trainNeuralOnOutcome(bool improved) {
        if (s_neuralSamples.empty()) return;
        ensureNeuralNetwork();
        if (improved) {
            // Bound the amount of work per record regardless of layer width.
            auto sampleBudget = std::clamp<size_t>(1280000u / s_neuralWeights.size(), 16, 256);
            auto stride = std::max<size_t>(1, s_neuralSamples.size() / sampleBudget);
            for (size_t i = 0; i < s_neuralSamples.size(); i += stride) {
                trainNeuralSample(s_neuralSamples[i], 0.003f);
            }
        } else {
            // Death is a negative outcome for the final local pattern; try the
            // opposite decisions nearby while the mutation search tests them.
            auto start = s_neuralSamples.size() > 12 ? s_neuralSamples.size() - 12 : 0;
            for (size_t i = start; i < s_neuralSamples.size(); ++i) {
                if (!s_neuralSamples[i].exploratory) continue;
                auto alternative = s_neuralSamples[i];
                alternative.hold = !alternative.hold;
                trainNeuralSample(alternative, 0.001f);
            }
        }
    }

    std::vector<FrameDecision> makeCandidate(
        std::vector<FrameDecision> const& champion,
        int deathFrame,
        int mutationWindow,
        int protectedFrame,
        uint32_t seed
    ) {
        auto candidate = champion;
        int memoryLimit = inputMemoryLimit();
        if ((int)candidate.size() > memoryLimit) candidate.resize(memoryLimit);
        deathFrame = std::clamp(deathFrame, 0, memoryLimit);
        if ((int)candidate.size() < deathFrame) candidate.resize(deathFrame, {false});
        protectedFrame = std::clamp(protectedFrame, 0, memoryLimit);
        // If the current attempt dies before the champion's record point,
        // mutating only after that point yields an empty range and repeats the
        // same failed pattern forever. Keep the champion itself intact, but
        // let this candidate explore the last window before its own death.
        int safePrefix = protectedFrame <= deathFrame ? protectedFrame : 0;
        int first = std::max(safePrefix, deathFrame - mutationWindow);
        int last = std::min(deathFrame - 1, (int)candidate.size() - 1);
        int span = last - first + 1;
        if (span <= 0) return candidate;

        auto nextRandom = [&seed]() {
            seed = seed * 747796405u + 2891336453u;
            uint32_t value = ((seed >> ((seed >> 28) + 4)) ^ seed) * 277803737u;
            return (value >> 22) ^ value;
        };
        int maxFlips = std::min(span, std::max(2, span / 12));
        int flips = 1 + (int)(nextRandom() % (uint32_t)maxFlips);
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

    std::string formatLevelPercent(float percent) {
        percent = std::clamp(percent, 0.0f, 100.0f);
        int tenths = (int)std::lround(percent * 10.0);
        tenths = std::clamp(tenths, 0, 1000);
        return std::to_string(tenths / 10) + "." + std::to_string(tenths % 10);
    }

    void refreshLevelBestLabel() {
        if (!s_levelBestLabel) return;
        auto text = "LEVEL BEST: " + formatLevelPercent(s_levelBestPercent) + "%";
        s_levelBestLabel->setString(text.c_str());
    }

    void bindLevelBest(GJGameLevel* level) {
        s_levelBestKey.clear();
        s_levelBestPercent = 0.0f;
        if (level) {
            int id = level->m_levelID;
            if (id > 0) {
                s_levelBestKey = "bot-level-best:" + std::to_string(id);
            } else {
                auto name = level->m_levelName.empty() ? std::string("unnamed") : level->m_levelName;
                s_levelBestKey = "bot-level-best:local:" + std::to_string(level->m_levelIndex) + ":" + name;
            }
            if (Mod::get()->hasSavedValue(s_levelBestKey)) {
                s_levelBestPercent = std::clamp(
                    (float)Mod::get()->getSavedValue<double>(s_levelBestKey), 0.0f, 100.0f);
            }
        }
        refreshLevelBestLabel();
    }

    // The bot's furthest percent on this level. Practice and normal runs share
    // one record, and it stays saved after leaving the level.
    void noteBotLevelProgress(float percent) {
        percent = std::clamp(percent, 0.0f, 100.0f);
        if (percent <= s_levelBestPercent + 0.049f) return;
        s_levelBestPercent = percent;
        refreshLevelBestLabel();
        if (!s_levelBestKey.empty()) {
            Mod::get()->setSavedValue(s_levelBestKey, (double)s_levelBestPercent);
        }
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
        drawPath(s_bestPracticeTrajectory, ccc4f(0.2f, 0.95f, 0.4f, 0.9f), ccc4f(0.45f, 1.0f, 0.55f, 1.0f));
        drawPath(s_segment.bestTrajectory, ccc4f(0.2f, 0.75f, 0.95f, 0.9f), ccc4f(0.45f, 0.85f, 1.0f, 1.0f));
        drawPath(s_currentTrajectory, ccc4f(0.15f, 0.8f, 1.0f, 0.8f), ccc4f(1.0f, 0.25f, 0.25f, 1.0f));
        if (!s_bestTrajectory.empty()) {
            s_trajectoryNode->drawDot(s_bestTrajectory.back().position, 6.0f,
                                      ccc4f(1.0f, 1.0f, 1.0f, 1.0f));
        }
        if (!s_bestPracticeTrajectory.empty()) {
            s_trajectoryNode->drawDot(s_bestPracticeTrajectory.back().position, 6.0f,
                                      ccc4f(0.2f, 1.0f, 0.4f, 1.0f));
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

    void drawHazardHitboxes() {
        if (!s_hazardNode) return;
        s_hazardNode->clear();
        if (!s_hazardNode->isVisible()) return;
        auto color = ccc4f(1.0f, 0.15f, 0.12f, 0.9f);
        for (auto const& rect : s_hazardRects) {
            auto left = rect.origin.x;
            auto bottom = rect.origin.y;
            auto right = left + rect.size.width;
            auto top = bottom + rect.size.height;
            s_hazardNode->drawSegment(ccp(left, bottom), ccp(right, bottom), 1.5f, color);
            s_hazardNode->drawSegment(ccp(right, bottom), ccp(right, top), 1.5f, color);
            s_hazardNode->drawSegment(ccp(right, top), ccp(left, top), 1.5f, color);
            s_hazardNode->drawSegment(ccp(left, top), ccp(left, bottom), 1.5f, color);
        }
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
        predictionSteps = std::clamp(predictionSteps, 5, 2500);
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
            ? s_segment.bestTrajectory
            : (s_practiceMode && !s_bestPracticeTrajectory.empty()
                ? s_bestPracticeTrajectory : s_bestTrajectory);
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
                if (drawn > 0) previous = previousMemory;
            }
        }

        // If the remembered route ends before the configured horizon, extend
        // the preview with the current motion estimate instead of truncating it.
        int memorySteps = 0;
        if (!memory.empty()) {
            auto currentPercent = s_currentTrajectory.back().percent;
            auto it = std::lower_bound(memory.begin(), memory.end(), currentPercent,
                [](TrailPoint const& point, float target) { return point.percent < target; });
            if (it != memory.end()) memorySteps = (int)std::distance(it, memory.end()) - 1;
        }
        for (int step = std::max(1, memorySteps + 1); step <= predictionSteps; ++step) {
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
        s_startDelayTicks = 0;
        bool hasChampion = s_practiceMode
            ? !s_bestPracticePath.empty() : !s_bestPath.empty();
        if (nextExploreRandom() % 100u < (hasChampion ? 15u : 20u)) {
            s_startDelayTicks = 12 + (int)(nextExploreRandom() % 49u);
        }
        s_allowPracticeMemoryGuess = !s_practiceMode &&
            !s_bestPracticePath.empty() && nextExploreRandom() % 100u < 70u;
    }

    bool nextNeuralHold(float dt, std::array<float, NEURAL_INPUTS> const& features) {
        s_exploreRemainingSeconds -= std::max(0.0f, dt);
        if (s_exploreRemainingSeconds <= 0.0f) {
            auto probability = neuralPrediction(features);
            auto draw = (float)(nextExploreRandom() % 10000u) / 10000.0f;
            s_exploreHold = draw < probability;
            // Keep decisions stable for natural 0.1–2.0 second intervals;
            // the network chooses the action, while this clock avoids jitter.
            s_exploreRemainingSeconds = 0.1f + (float)(nextExploreRandom() % 1901u) / 1000.0f;
        }
        return s_exploreHold;
    }

    size_t frameAtPercent(std::vector<TrailPoint> const& trajectory, float percent) {
        if (trajectory.empty()) return 0;
        auto it = std::lower_bound(
            trajectory.begin(), trajectory.end(), percent,
            [](TrailPoint const& point, float target) { return point.percent < target; }
        );
        return (size_t)std::distance(trajectory.begin(), it);
    }

    size_t bestFrameAtPercent(float percent) {
        return frameAtPercent(s_bestTrajectory, percent);
    }

    bool guessFromSimilarMemory(
        CCPoint position,
        std::vector<FrameDecision> const& inputs,
        std::vector<TrailPoint> const& trajectory,
        bool& guessedHold,
        float maxDistance = 96.0f
    ) {
        auto count = std::min(inputs.size(), trajectory.size());
        if (count < 2) return false;

        float closestDistanceSquared = maxDistance * maxDistance;
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
        s_neuralSamples.clear();
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
        BotState::s_practiceMode = this->m_isPracticeMode;
        BotState::s_currentFrame = 0;
        BotState::s_attemptsCount = 0;
        BotState::s_postDeathCooldown = 0.0f;
        BotState::s_lastDeathPercent = -1000.0f;
        BotState::s_sameSpotDeaths = 0;
        BotState::s_normalScore = 0;
        BotState::s_practiceScore = 0;
        BotState::s_bestDistanceX = 0.0f;
        BotState::s_bestPracticeDistanceX = 0.0f;
        BotState::s_bestPath.clear();
        BotState::s_candidatePath.clear();
        BotState::s_bestPracticePath.clear();
        BotState::s_candidatePracticePath.clear();
        BotState::s_bestTrajectory.clear();
        BotState::s_bestPracticeTrajectory.clear();
        BotState::s_hazardRects.clear();
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
        BotState::ensureNeuralNetwork();
        BotState::s_levelActiveSeconds = 0.0f;
        BotState::s_levelTimerRefresh = 0.0f;

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

        auto legend = CCLabelBMFont::create("TRAIL: red=click gold=normal-best green=practice-best cyan=current purple=prediction", "chatFont.fnt");
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

        auto deathAnalysis = CCLabelBMFont::create("DEATH ANALYSIS: waiting for a death", "chatFont.fnt");
        deathAnalysis->setScale(0.34f);
        deathAnalysis->setAnchorPoint(ccp(0.0f, 0.5f));
        deathAnalysis->setPosition(ccp(12.0f, -39.0f));
        hudNode->addChild(deathAnalysis);
        BotState::s_deathAnalysisLabel = deathAnalysis;

        auto neuralTitle = CCLabelBMFont::create("NEURAL ACTIVITY", "chatFont.fnt");
        neuralTitle->setScale(0.42f);
        neuralTitle->setAnchorPoint(ccp(1.0f, 0.5f));
        neuralTitle->setPosition(ccp(winSize.width - 12.0f, 0.0f));
        hudNode->addChild(neuralTitle);

        auto neuralMonitor = CCLabelBMFont::create("SPIKES: 0 / 0   SYNAPSES: 0 / 0   MEAN: 0%", "chatFont.fnt");
        neuralMonitor->setScale(0.34f);
        neuralMonitor->setAnchorPoint(ccp(1.0f, 0.5f));
        neuralMonitor->setPosition(ccp(winSize.width - 12.0f, -12.0f));
        hudNode->addChild(neuralMonitor);
        BotState::s_neuralMonitorLabel = neuralMonitor;

        auto levelTimer = CCLabelBMFont::create("BOT ACTIVE: 00:00", "chatFont.fnt");
        levelTimer->setScale(0.38f);
        levelTimer->setAnchorPoint(ccp(1.0f, 0.5f));
        levelTimer->setPosition(ccp(winSize.width - 12.0f, -25.0f));
        hudNode->addChild(levelTimer);
        BotState::s_levelTimeLabel = levelTimer;

        auto levelBest = CCLabelBMFont::create("LEVEL BEST: 0.0%", "chatFont.fnt");
        levelBest->setScale(0.38f);
        levelBest->setAnchorPoint(ccp(1.0f, 0.5f));
        levelBest->setPosition(ccp(winSize.width - 12.0f, -38.0f));
        levelBest->setColor(ccc3(255, 214, 90));
        hudNode->addChild(levelBest);
        BotState::s_levelBestLabel = levelBest;
        BotState::bindLevelBest(level);

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

        auto hazards = CCDrawNode::create();
        hazards->setID("ai-bot-hazard-hitboxes"_spr);
        hazards->setVisible(Mod::get()->getSettingValue<bool>("show-hazards"));
        BotState::s_hazardNode = hazards;
        if (this->m_objectLayer) this->m_objectLayer->addChild(hazards, 1001);
        BotState::beginTrajectoryAttempt();

        this->addChild(hudNode);
        BotState::refreshIndicator();

        return true;
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        BotState::s_practiceMode = this->m_isPracticeMode;
        BotState::s_currentFrame = 0;
        BotState::s_lastHold = false;
        BotState::s_cpsCooldown = 0.0f;
        BotState::s_naturalClickCooldownTicks = 0;
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
            auto const& practiceSource = BotState::s_candidatePracticePath.empty()
                ? BotState::s_bestPracticePath : BotState::s_candidatePracticePath;
            auto const& normalSource = BotState::s_candidatePath.empty()
                ? BotState::s_bestPath : BotState::s_candidatePath;
            if (this->m_isPracticeMode && !practiceSource.empty()) {
                BotState::s_currentAttempt = practiceSource;
            } else {
                BotState::s_currentAttempt = normalSource;
            }
        }
        this->handleButton(false, 1, true);
        if (this->m_player2) this->handleButton(false, 2, true);
        BotState::beginTrajectoryAttempt();
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        BotState::s_practiceMode = this->m_isPracticeMode;

        // Keep the HUD in sync when the bot is toggled from the pause menu.
        BotState::refreshIndicator();

        if (BotState::s_enabled && dt > 0.0f) {
            BotState::s_levelActiveSeconds += dt;
            BotState::s_levelTimerRefresh += dt;
            if (BotState::s_levelTimeLabel && BotState::s_levelTimerRefresh >= 0.25f) {
                BotState::s_levelTimerRefresh = 0.0f;
                auto totalSeconds = (int)BotState::s_levelActiveSeconds;
                auto timeText = "BOT ACTIVE: " + std::to_string(totalSeconds / 60) + ":" +
                    (totalSeconds % 60 < 10 ? "0" : "") + std::to_string(totalSeconds % 60);
                BotState::s_levelTimeLabel->setString(timeText.c_str());
            }
        }

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
            BotState::drawHazardHitboxes();
            BotState::drawPredictedTrajectory();
            return;
        }

        BotState::noteBotLevelProgress(progress);

        if (BotState::s_postDeathCooldown > 0.0f) {
            BotState::s_postDeathCooldown = std::max(0.0f, BotState::s_postDeathCooldown - dt);
            BotState::s_lastHold = false;
            BotState::drawPlayerHitbox(playerRect);
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
                if (progress > BotState::s_bestPracticeDistanceX) {
                    BotState::s_bestPracticeDistanceX = progress;
                    BotState::s_bestPracticePath = BotState::s_currentAttempt;
                    BotState::s_bestPracticeTrajectory = BotState::s_currentTrajectory;
                }
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
        bool neuralExploration = false;
        if (frame < (int)knownLength) {
            hold = inputPath[frame].hold;
        } else {
            // The old path ends at the best run's frontier. Explore beyond it
            // by first looking for a similar remembered position and reusing
            // its next click decision. Use a varied pattern when memory has no
            // nearby match. Decisions are retained up to the tick limit.
            auto features = BotState::neuralFeatures(position, progress);
            hold = BotState::nextNeuralHold(dt, features);
            bool startDelayActive = frame < BotState::s_startDelayTicks;
            if (startDelayActive) hold = false;
            neuralExploration = !startDelayActive;
            auto const& guessInputs = BotState::s_trainingSegment && !BotState::s_segment.bestInputs.empty()
                ? BotState::s_segment.bestInputs
                : (this->m_isPracticeMode && !BotState::s_bestPracticePath.empty()
                    ? BotState::s_bestPracticePath : BotState::s_bestPath);
            auto const& guessTrajectory = BotState::s_trainingSegment && !BotState::s_segment.bestTrajectory.empty()
                ? BotState::s_segment.bestTrajectory
                : (this->m_isPracticeMode && !BotState::s_bestPracticeTrajectory.empty()
                    ? BotState::s_bestPracticeTrajectory : BotState::s_bestTrajectory);
            bool rememberedGuess = false;
            bool usedPracticeGuess = false;
            if (!startDelayActive && BotState::s_allowPracticeMemoryGuess) {
                usedPracticeGuess = BotState::guessFromSimilarMemory(
                    position, BotState::s_bestPracticePath,
                    BotState::s_bestPracticeTrajectory, rememberedGuess, 160.0f);
            }
            if (!startDelayActive && !usedPracticeGuess &&
                !BotState::guessFromSimilarMemory(position, guessInputs, guessTrajectory, rememberedGuess)) {
                rememberedGuess = hold;
            }
            if (!startDelayActive) hold = rememberedGuess;
            if (canStoreInput && frame < (int)inputPath.size()) inputPath[frame].hold = hold;
        }

        if (frame < BotState::s_startDelayTicks) {
            hold = false;
            neuralExploration = false;
        }

        if (!BotState::s_trainingSegment && !BotState::s_segment.bestInputs.empty()) {
            if (!BotState::s_segmentPlaybackActive && !BotState::s_segmentPlaybackDone && progress >= BotState::s_segment.startPercent) {
                BotState::s_segmentPlaybackActive = true;
                BotState::s_segmentPlaybackFrame = 0;
            }
            if (BotState::s_segmentPlaybackActive) {
                if (BotState::s_segmentPlaybackFrame < (int)BotState::s_segment.bestInputs.size()) {
                    hold = BotState::s_segment.bestInputs[BotState::s_segmentPlaybackFrame++].hold;
                    neuralExploration = false;
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
        if (BotState::s_naturalClickCooldownTicks > 0) {
            --BotState::s_naturalClickCooldownTicks;
        }
        if (hold && !BotState::s_lastHold &&
            (BotState::s_cpsCooldown > 0.0f || BotState::s_naturalClickCooldownTicks > 0)) {
            hold = false;
        }
        bool click = hold && !BotState::s_lastHold;
        if (click) {
            BotState::s_cpsCooldown = 1.0f / cpsTarget;
            // Add natural variation to rapid tap spacing: wait 12–18 game ticks.
            BotState::s_naturalClickCooldownTicks = 12 + (int)(BotState::nextExploreRandom() % 7u);
        }
        if (hold != BotState::s_lastHold) {
            this->handleButton(hold, 1, true);
            if (this->m_player2) this->handleButton(hold, 2, true);
        }
        BotState::s_lastHold = hold;

        auto neuralSampleFeatures = BotState::neuralFeatures(position, progress);
        if (BotState::s_neuralSamples.size() < (size_t)memoryLimit) {
            BotState::s_neuralSamples.push_back({neuralSampleFeatures, hold, neuralExploration});
        }
        BotState::recordTrajectoryPoint(position, click, progress);

        BotState::drawPlayerHitbox(playerRect);
        BotState::drawHazardHitboxes();
        BotState::drawPredictedTrajectory();
        if (frame < (int)inputPath.size()) inputPath[frame].hold = hold;
        BotState::s_currentFrame++;
    }

    void levelComplete() {
        PlayLayer::levelComplete();
        if (BotState::s_enabled) {
            BotState::noteBotLevelProgress(std::max(this->getCurrentPercent(), 100.0f));
        }
    }

    void onExit() {
        // These nodes belong to this PlayLayer. Clear the cached pointers
        // before the layer releases its children so setting listeners cannot
        // touch freed UI objects after leaving a level.
        BotState::s_indicatorDot = nullptr;
        BotState::s_indicatorLabel = nullptr;
        BotState::s_trajectoryLegend = nullptr;
        BotState::s_scoreLabel = nullptr;
        BotState::s_deathAnalysisLabel = nullptr;
        BotState::s_neuralMonitorLabel = nullptr;
        BotState::s_levelTimeLabel = nullptr;
        BotState::s_levelBestLabel = nullptr;
        BotState::s_trajectoryNode = nullptr;
        BotState::s_hitboxNode = nullptr;
        BotState::s_hazardNode = nullptr;
        BotState::s_hazardRects.clear();
        BotState::s_predictionNode = nullptr;
        BotState::s_hasLastTrajectoryPoint = false;
        PlayLayer::onExit();
    }
};

// Cache the collision rectangles Geometry Dash checks for spikes, saws and
// other lethal objects around player one. The drawing and learner can then
// use the same geometry that the game uses for collision checks.
class $modify(AIBotBaseGameLayer, GJBaseGameLayer) {
    void collisionCheckObjects(PlayerObject* player, std::vector<GameObject*>* objects, int count, float dt) {
        auto play = PlayLayer::get();
        if (play && player == play->m_player1) {
            BotState::s_hazardRects.clear();
            if (objects) {
                for (int i = 0; i < count; ++i) {
                    auto* object = (*objects)[i];
                    if (!object) continue;
                    auto type = object->m_objectType;
                    if (type == GameObjectType::Hazard || type == GameObjectType::AnimatedHazard) {
                        BotState::s_hazardRects.push_back(object->getObjectRect());
                    }
                }
            }
        }
        GJBaseGameLayer::collisionCheckObjects(player, objects, count, dt);
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
        if (pl && pl->m_isPracticeMode) {
            // Skip the delay on some deaths so Practice learning can continue
            // immediately; otherwise use the configured random delay range.
            bool skipDelay = BotState::nextExploreRandom() % 100u < 20u;
            if (skipDelay) {
                BotState::s_postDeathCooldown = 0.0f;
            } else {
                auto maxDelayMs = (int)Mod::get()->getSettingValue<int64_t>("practice-death-input-delay-ms");
                maxDelayMs = std::clamp(maxDelayMs, 4, 2000);
                auto delayMs = 4 + (int)(BotState::nextExploreRandom() % (uint32_t)(maxDelayMs - 3));
                BotState::s_postDeathCooldown = delayMs / 1000.0f;
            }
            pl->handleButton(false, 1, true);
            if (pl->m_player2) pl->handleButton(false, 2, true);
            BotState::s_lastHold = false;
            BotState::s_cpsCooldown = 0.0f;
            BotState::s_naturalClickCooldownTicks = 0;
        }
        float currentPercent = pl ? pl->getCurrentPercent() : 0.0f;
        BotState::noteBotLevelProgress(currentPercent);
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
        bool practiceMode = pl && pl->m_isPracticeMode;
        float previousBest = BotState::s_trainingSegment
            ? BotState::s_segment.bestPercent
            : (practiceMode ? BotState::s_bestPracticeDistanceX : BotState::s_bestDistanceX);
        bool improved = currentPercent > previousBest;
        BotState::trainNeuralOnOutcome(improved);
        double& score = practiceMode ? BotState::s_practiceScore : BotState::s_normalScore;
        auto deathPenalty = (int)Mod::get()->getSettingValue<int64_t>("learning-death-penalty");
        deathPenalty = std::clamp(deathPenalty, 0, 1000000);
        score = std::clamp(score - deathPenalty, -BotState::s_scoreLimit, BotState::s_scoreLimit);
        BotState::updateScoreLabel();
        int negativeScoreMutation = score < 0 ? std::min(120, (int)(-score / 2.0)) : 0;
        mutationWindow = std::min(120, mutationWindow + negativeScoreMutation);

        if (BotState::s_deathAnalysisLabel) {
            auto analysis = "DEATH " + std::to_string(std::round(currentPercent * 10.0f) / 10.0f) +
                "% | SAME SPOT " + std::to_string(BotState::s_sameSpotDeaths) +
                " | HAZARDS " + std::to_string(BotState::s_hazardRects.size()) +
                " | MUTATION " + std::to_string(mutationWindow) + " TICKS";
            BotState::s_deathAnalysisLabel->setString(analysis.c_str());
        }

        if (BotState::s_trainingSegment) {
            if (improved) {
                BotState::s_segment.bestPercent = currentPercent;
                BotState::s_segment.bestInputs = BotState::s_segmentAttempt;
                BotState::s_segment.bestTrajectory = BotState::s_currentTrajectory;
                BotState::drawStoredTrajectory();
            }
            auto const& champion = BotState::s_segment.bestInputs.empty()
                ? BotState::s_segmentAttempt : BotState::s_segment.bestInputs;
            auto protectedFrame = (int)BotState::frameAtPercent(
                BotState::s_segment.bestTrajectory, BotState::s_segment.bestPercent);
            BotState::s_segmentCandidate = BotState::makeCandidate(
                champion, deathFrame, mutationWindow, protectedFrame,
                (uint32_t)(BotState::s_attemptsCount * 314159u + deathFrame)
            );

        }
        else if (practiceMode) {
            if (improved) {
                BotState::s_bestPracticeDistanceX = currentPercent;
                BotState::s_bestPracticePath = BotState::s_currentAttempt;
                BotState::s_bestPracticeTrajectory = BotState::s_currentTrajectory;
                BotState::drawStoredTrajectory();
            }
            auto const& champion = BotState::s_bestPracticePath.empty()
                ? BotState::s_currentAttempt : BotState::s_bestPracticePath;
            auto protectedFrame = (int)BotState::frameAtPercent(
                BotState::s_bestPracticeTrajectory, BotState::s_bestPracticeDistanceX);
            BotState::s_candidatePracticePath = BotState::makeCandidate(
                champion, deathFrame, mutationWindow, protectedFrame,
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
            auto protectedFrame = (int)BotState::bestFrameAtPercent(BotState::s_bestDistanceX);
            BotState::s_candidatePath = BotState::makeCandidate(
                champion, deathFrame, mutationWindow, protectedFrame,
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
        if (auto playLayer = PlayLayer::get()) {
            BotState::s_practiceMode = playLayer->m_isPracticeMode;
        }
        BotState::drawStoredTrajectory();
        if (auto playLayer = PlayLayer::get(); playLayer && playLayer->m_player1) {
            BotState::drawPlayerHitbox(playLayer->m_player1->getObjectRect());
            BotState::drawHazardHitboxes();
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
    listenForSettingChanges<bool>("show-hazards", [](bool visible) {
        if (BotState::s_hazardNode) {
            BotState::s_hazardNode->setVisible(visible);
            if (visible) BotState::drawHazardHitboxes();
        }
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
