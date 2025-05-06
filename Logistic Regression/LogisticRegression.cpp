#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

using namespace std;

class LogisticRegression {
public:
	vector<double> weights;
	double bias = 0.0;
	double learning_rate = 0.1;
	int max_iter = 20000;
	double tol = 1e-6;

	LogisticRegression(int feature_size, double lr = 0.1, int max_it = 20000, double tolerance = 1e-6)
		: learning_rate(lr), max_iter(max_it), tol(tolerance) {
		weights = vector<double>(feature_size, 0.01);
		bias = 0.0;
	}

	static double sigmoid(double x) {
		return 1.0 / (1.0 + exp(-x));
	}

	double compute_cost(const vector<vector<double>>& x, const vector<double>& y) {
		double cost = 0.0;
		for (int i = 0; i < x.size(); i++) {
			double z = bias;
			for (int j = 0; j < weights.size(); j++) {
				z += weights[j] * x[i][j];
			}
			double pred = sigmoid(z);
			cost += -y[i] * log(pred) - (1 - y[i]) * log(1 - pred);
		}
		return cost / y.size();
	}

	void compute_gradient(const vector<vector<double>>& x, const vector<double>& y, vector<double>& w_grad, double& b_grad) {
		int n = x.size();
		int d = weights.size();
		w_grad.assign(d, 0.0);
		b_grad = 0.0;

		for (int i = 0; i < n; i++) {
			double z = bias;
			for (int j = 0; j < d; j++) z += weights[j] * x[i][j];
			double pred = sigmoid(z);
			double diff = pred - y[i];

			for (int j = 0; j < d; j++) {
				w_grad[j] += x[i][j] * diff;
			}
			b_grad += diff;
		}

		for (int j = 0; j < d; j++) w_grad[j] /= n;
		b_grad /= n;
	}

	void fit(const vector<vector<double>>& x, const vector<double>& y) {
		double prev_cost = compute_cost(x, y);

		for (int i = 0; i < max_iter; i++) {
			vector<double> grad_w(weights.size(), 0.0);
			double grad_b = 0.0;
			compute_gradient(x, y, grad_w, grad_b);

			for (int j = 0; j < weights.size(); j++)
				weights[j] -= learning_rate * grad_w[j];
			bias -= learning_rate * grad_b;

			double cost = compute_cost(x, y);
			if (i % 1000 == 0)
				cout << "Epoch: " << i << " Cost: " << cost << endl;

			if (abs(prev_cost - cost) < tol) {
				cout << "Early stopping at iteration " << i << ", cost change: " << abs(prev_cost - cost) << endl;
				break;
			}
			prev_cost = cost;
		}
	}

	vector<double> predict_proba(const vector<vector<double>>& x) {
		vector<double> probs;
		for (const auto& row : x) {
			double z = bias;
			for (int j = 0; j < weights.size(); j++) {
				z += weights[j] * row[j];
			}
			probs.push_back(sigmoid(z));
		}
		return probs;
	}

	vector<int> predict(const vector<vector<double>>& x, double threshold = 0.5) {
		vector<double> probs = predict_proba(x);
		vector<int> preds;
		for (double p : probs) {
			preds.push_back(p >= threshold ? 1 : 0);
		}
		return preds;
	}
};


struct DataRecord {
	double Age, Weight, BloodSugar, Gender, Diabetes;
};

int loadcsvfile2(string fname, map<string, vector<double>>& datas) {
	ifstream inputFile(fname);
	if (!inputFile.is_open()) {
		cerr << "Could not open the file" << endl;
		return 0;
	}

	string line;
	getline(inputFile, line); // Skip header
	while (getline(inputFile, line)) {
		stringstream ss(line);
		string token;
		DataRecord record;

		getline(ss, token, ','); record.Age = stod(token);
		getline(ss, token, ','); record.Weight = stod(token);
		getline(ss, token, ','); record.BloodSugar = stod(token);
		getline(ss, token, ','); record.Gender = (token == "M") ? 1 : 0;
		getline(ss, token, ','); record.Diabetes = stod(token);

		datas["Age"].push_back(record.Age);
		datas["Weight"].push_back(record.Weight);
		datas["BloodSugar"].push_back(record.BloodSugar);
		datas["Gender"].push_back(record.Gender);
		datas["Diabetes"].push_back(record.Diabetes);
	}

	return 1;
}

double mean(const vector<double>& data) {
	double sum = 0.0;
	for (double v : data) sum += v;
	return sum / data.size();
}

double stddev(const vector<double>& data) {
	double m = mean(data), sum = 0.0;
	for (double v : data) sum += (v - m) * (v - m);
	return sqrt(sum / data.size());
}

vector<double> featureScaling(const vector<double>& data) {
	double m = mean(data), sd = stddev(data);
	vector<double> scaled(data.size());
	for (int i = 0; i < data.size(); i++)
		scaled[i] = (data[i] - m) / sd;
	return scaled;
}

int main() {
	map<string, vector<double>> datas;
	if (!loadcsvfile2("Diabetes_Data.csv", datas)) return -1;

	datas["Age"] = featureScaling(datas["Age"]);
	datas["Weight"] = featureScaling(datas["Weight"]);
	datas["BloodSugar"] = featureScaling(datas["BloodSugar"]);

	vector<vector<double>> x;
	for (int i = 0; i < datas["Age"].size(); i++) {
		x.push_back({
			datas["Age"][i],
			datas["Weight"][i],
			datas["BloodSugar"][i],
			datas["Gender"][i]
			});
	}

	vector<double> y = datas["Diabetes"];

	LogisticRegression model(4, 0.1, 20000, 1e-6);
	model.fit(x, y);

	auto predictions = model.predict(x);
	cout << "Sample predictions:\n";
	for (int i = 0; i < 5; i++) {
		cout << "Actual: " << y[i] << ", Predicted: " << predictions[i] << endl;
	}

	cin.get();
	return 0;
}
