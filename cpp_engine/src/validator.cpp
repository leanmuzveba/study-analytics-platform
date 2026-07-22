#include "validator.hpp"
#include <regex>
#include <cstdlib>

namespace studytok {

bool Validator::isValidDate(const std::string& date) {
    static const std::regex pattern(R"(^\d{4}-\d{2}-\d{2}$)");
    return std::regex_match(date, pattern);
}

bool Validator::parseDouble(const std::string& s, double& out) {
    if (s.empty()) return false;
    try {
        size_t pos;
        out = std::stod(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

bool Validator::parseInt(const std::string& s, int& out) {
    if (s.empty()) return false;
    try {
        size_t pos;
        out = std::stoi(s, &pos);
        return pos == s.size();
    } catch (...) {
        return false;
    }
}

bool Validator::inRange(double value, double lo, double hi) {
    return value >= lo && value <= hi;
}

std::optional<StudyLogRow> Validator::validate(
    const std::vector<std::string>& f, std::string& reasonOut) {

    // Expected columns:
    // student_id,date,subject,study_hours,active_recall_score,
    // rest_hours,sessions_count,distraction_events,exam_score
    if (f.size() != 9) {
        reasonOut = "expected 9 columns, got " + std::to_string(f.size());
        return std::nullopt;
    }

    if (f[0].empty()) {
        reasonOut = "missing student_id";
        return std::nullopt;
    }
    if (!isValidDate(f[1])) {
        reasonOut = "invalid date format: " + f[1];
        return std::nullopt;
    }
    if (f[2].empty()) {
        reasonOut = "missing subject";
        return std::nullopt;
    }

    StudyLogRow row;
    row.student_id = f[0];
    row.date = f[1];
    row.subject = f[2];

    double studyHours;
    if (!parseDouble(f[3], studyHours) || !inRange(studyHours, 0.0, 24.0)) {
        reasonOut = "study_hours out of range or invalid: " + f[3];
        return std::nullopt;
    }
    row.study_hours = studyHours;

    double recall;
    if (!parseDouble(f[4], recall) || !inRange(recall, 0.0, 100.0)) {
        reasonOut = "active_recall_score out of range or invalid: " + f[4];
        return std::nullopt;
    }
    row.active_recall_score = recall;

    double rest;
    if (!parseDouble(f[5], rest) || !inRange(rest, 0.0, 24.0)) {
        reasonOut = "rest_hours out of range or invalid: " + f[5];
        return std::nullopt;
    }
    row.rest_hours = rest;

    int sessions;
    if (!parseInt(f[6], sessions) || sessions < 0) {
        reasonOut = "sessions_count invalid: " + f[6];
        return std::nullopt;
    }
    row.sessions_count = sessions;

    int distractions;
    if (!parseInt(f[7], distractions) || distractions < 0) {
        reasonOut = "distraction_events invalid: " + f[7];
        return std::nullopt;
    }
    row.distraction_events = distractions;

    // exam_score is optional — blank means "awaiting prediction"
    if (!f[8].empty()) {
        double examScore;
        if (!parseDouble(f[8], examScore) || !inRange(examScore, 0.0, 100.0)) {
            reasonOut = "exam_score out of range or invalid: " + f[8];
            return std::nullopt;
        }
        row.exam_score = examScore;
    }

    return row;
}

} // namespace studytok
