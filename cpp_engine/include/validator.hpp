#pragma once

#include "csv_parser.hpp"
#include <string>
#include <optional>

namespace studytok {

// Validates raw string fields for a single CSV row and, if all checks pass,
// produces a populated StudyLogRow. If validation fails, returns nullopt
// and writes a human-readable reason into `reasonOut`.
class Validator {
public:
    static std::optional<StudyLogRow> validate(
        const std::vector<std::string>& fields,
        std::string& reasonOut);

private:
    static bool isValidDate(const std::string& date);
    static bool parseDouble(const std::string& s, double& out);
    static bool parseInt(const std::string& s, int& out);
    static bool inRange(double value, double lo, double hi);
};

} // namespace studytok
