#include "csv_parser.hpp"
#include "validator.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

namespace studytok {

std::string StudyLogRow::toCsvLine() const {
    std::ostringstream oss;
    oss << student_id << ','
        << date << ','
        << subject << ','
        << study_hours << ','
        << active_recall_score << ','
        << rest_hours << ','
        << sessions_count << ','
        << distraction_events << ',';
    if (exam_score.has_value()) {
        oss << *exam_score;
    }
    return oss.str();
}

// Simple CSV line splitter. Handles plain comma-separated fields.
// (No embedded-comma/quote support — raw logs for this pipeline are
// generated, not free-text, so this keeps the engine fast and simple.)
std::vector<std::string> CSVParser::splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream ss(line);
    while (std::getline(ss, field, ',')) {
        fields.push_back(field);
    }
    // handle trailing empty field (e.g. line ends with ",")
    if (!line.empty() && line.back() == ',') {
        fields.push_back("");
    }
    return fields;
}

ParseResult CSVParser::parseFile(const std::string& rawPath) {
    ParseResult result;
    std::ifstream in(rawPath);
    if (!in.is_open()) {
        result.rejectionReasons.push_back("could not open file: " + rawPath);
        return result;
    }

    std::string line;
    bool isHeader = true;

    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (isHeader) {
            isHeader = false;
            continue; // skip header row
        }

        result.totalRowsRead++;
        auto fields = splitCsvLine(line);

        std::string reason;
        auto row = Validator::validate(fields, reason);
        if (row.has_value()) {
            result.validRows.push_back(*row);
        } else {
            result.rejectedRows++;
            result.rejectionReasons.push_back(
                "row " + std::to_string(result.totalRowsRead) + ": " + reason);
        }
    }

    return result;
}

bool CSVParser::writeCleanFile(const std::string& cleanPath,
                                const std::vector<StudyLogRow>& rows) {
    std::ofstream out(cleanPath);
    if (!out.is_open()) return false;

    out << "student_id,date,subject,study_hours,active_recall_score,"
           "rest_hours,sessions_count,distraction_events,exam_score\n";
    for (const auto& row : rows) {
        out << row.toCsvLine() << "\n";
    }
    return true;
}

} // namespace studytok
