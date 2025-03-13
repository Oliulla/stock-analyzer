#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp> 

using json = nlohmann::json;
using namespace std;

int main() {
    ifstream file("../data-fetcher/stock_data.json");  
    if (!file) {
        cerr << "Error opening stock_data.json" << endl;
        return 1;
    }

    json data;
    file >> data;  // Read the JSON file
    file.close();

    // Extract stock details
    string symbol = data["Global Quote"]["01. symbol"];
    string price = data["Global Quote"]["05. price"];
    string change = data["Global Quote"]["09. change"];
    string volume = data["Global Quote"]["06. volume"];

    // Display extracted information
    cout << "Stock Symbol: " << symbol << endl;
    cout << "Current Price: $" << price << endl;
    cout << "Change: " << change << endl;
    cout << "Volume: " << volume << " shares" << endl;

    return 0;
}
