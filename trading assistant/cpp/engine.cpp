#include <cmath>

extern "C" {

double calculate_volatility(double* prices, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; i++)
        mean += prices[i];
    mean /= n;

    double variance = 0.0;
    for (int i = 0; i < n; i++)
        variance += (prices[i] - mean) * (prices[i] - mean);

    return std::sqrt(variance / n);
}

int find_support_resistance(double* prices, int n,
                            double* supports,
                            double* resistances) {
    int s = 0, r = 0;
    for (int i = 1; i < n - 1; i++) {
        if (prices[i] < prices[i - 1] && prices[i] < prices[i + 1])
            supports[s++] = prices[i];
        if (prices[i] > prices[i - 1] && prices[i] > prices[i + 1])
            resistances[r++] = prices[i];
    }
    return s * 1000 + r;
}
double calculate_sma(double* prices, int n) {
    double sum = 0;
    for(int i=0;i<n;i++) sum += prices[i];
    return sum / n;
}
double calculate_ema(double* prices, int n, double alpha) {
    double ema = prices[0];
    for(int i=1;i<n;i++) {
        ema = alpha*prices[i] + (1-alpha)*ema;
    }
    return ema;
}
double calculate_rsi(double* prices, int n) {
    double gain = 0, loss = 0;
    for(int i=1;i<n;i++) {
        double diff = prices[i] - prices[i-1];
        if(diff>0) gain += diff;
        else loss -= diff;
    }
    if(loss==0) return 100.0;
    double rs = gain/loss;
    return 100 - (100/(1+rs));
}
}
