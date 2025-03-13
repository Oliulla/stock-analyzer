#include <stdio.h>
#include <curl/curl.h>
#include <time.h>

// Log function
void log_message(const char *message) {
    FILE *log_file = fopen("fetcher.log", "a");
    if (log_file) {
        time_t now;
        time(&now);
        fprintf(log_file, "%s - %s\n", ctime(&now), message);
        fclose(log_file);
    }
}

size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    return fwrite(ptr, size, nmemb, stream);
}

int main() {
    CURL *curl;
    FILE *fp;
    CURLcode res;
    char *url = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=26CA28RXCQ8HIK15";
    char outfilename[] = "stock_data.json";

    curl = curl_easy_init();
    if (curl) {
        fp = fopen(outfilename, "wb");
        curl_easy_setopt(curl, CURLOPT_URL, url);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        res = curl_easy_perform(curl);
        curl_easy_cleanup(curl);
        fclose(fp);
        
        if (res == CURLE_OK) {
            log_message("Successfully fetched stock data and saved to stock_data.json");
        } else {
            log_message("Error fetching stock data");
        }
    } else {
        log_message("Error initializing cURL");
    }

    printf("Stock data saved to %s\n", outfilename);
    return 0;
}
