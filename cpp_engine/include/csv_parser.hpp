#pragma once

#include <string>
#include <vector>
#include <optional>

namespace studytok {

// One row of a student study log.
// exam_score is optional: rows awaiting prediction won't have it yet.
struct StudyLogRow {
    std::string student_id;
    std::string date;              // YYYY-MM-DD
    std::string subject;
    double study_hours = 0.0;
    double active_recall_score = 0.0;
    double rest_hours = 0.0;
    int sessions_count = 0;
    int distraction_events = 0;
    std::optional<double> exam_score;

    std::string toCsvLine() const;
};

// Result of a parse+validate pass over a CSV file.
struct ParseResult {
    std::vector<StudyLogRow> validRows;
    std::size_t totalRowsRead = 0;
    std::size_t rejectedRows = 0;
    std::vector<std::string> rejectionReasons; // one entry per rejected row
};

class CSVParser {
public:
    // Parses and validates rawPath. Does not write anything.
    static ParseResult parseFile(const std::string& rawPath);

    // Writes validRows from a ParseResult out to cleanPath as CSV
    // (with header). Returns true on success.
    static bool writeCleanFile(const std::string& cleanPath,
                                const std::vector<StudyLogRow>& rows);

private:
    static std::vector<std::string> splitCsvLine(const std::string& line);
};

} // namespace studytok
