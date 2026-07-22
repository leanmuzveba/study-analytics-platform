#include "csv_parser.hpp"
#include <iostream>

// Usage: studytok_engine <raw_input.csv> <clean_output.csv>
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <raw_input.csv> <clean_output.csv>\n";
        return 1;
    }

    std::string rawPath = argv[1];
    std::string cleanPath = argv[2];

    auto result = studytok::CSVParser::parseFile(rawPath);

    if (!studytok::CSVParser::writeCleanFile(cleanPath, result.validRows)) {
        std::cerr << "Error: could not write output file " << cleanPath << "\n";
        return 1;
    }

    std::cout << "StudyTok CSV Engine — parse summary\n";
    std::cout << "  Input file:     " << rawPath << "\n";
    std::cout << "  Output file:    " << cleanPath << "\n";
    std::cout << "  Rows read:      " << result.totalRowsRead << "\n";
    std::cout << "  Rows accepted:  " << result.validRows.size() << "\n";
    std::cout << "  Rows rejected:  " << result.rejectedRows << "\n";

    if (result.rejectedRows > 0) {
        std::cout << "  Rejection details:\n";
        for (const auto& reason : result.rejectionReasons) {
            std::cout << "    - " << reason << "\n";
        }
    }

    return 0;
}
