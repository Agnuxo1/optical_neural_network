#include "csv_loader.hpp"
#include "optical_model.hpp"
#include "training.hpp"
#include "utils.hpp"
#include <iostream>
#include <string>

struct Cmd {
  std::string train_csv = "data/train.csv";
  std::string test_csv  = "data/test.csv";
  std::string submission = "submission.csv";
  int epochs = 2000;
  int batch  = 512;
  float lr   = 2e-3f;
};

static Cmd parse_cmd(int argc, char** argv) {
  Cmd c;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) { std::cerr << "Missing value for " << name << "\n"; std::exit(1); }
      return std::string(argv[++i]);
    };
    if      (a == std::string("--train")) c.train_csv = next("--train");
    else if (a == std::string("--test")) c.test_csv = next("--test");
    else if (a == std::string("--submission")) c.submission = next("--submission");
    else if (a == std::string("--epochs")) c.epochs = std::stoi(next("--epochs"));
    else if (a == std::string("--batch"))  c.batch  = std::stoi(next("--batch"));
    else if (a == std::string("--lr"))     c.lr     = std::stof(next("--lr"));
    else std::cerr << "Unknown arg: " << a << "\n";
  }
  return c;
}

int main(int argc, char** argv) {
  try {
    Cmd cmd = parse_cmd(argc, argv);
    utils::set_seed(1337u);

    std::cout << "[INFO] Loading train: " << cmd.train_csv << "\n";
    TrainSet train = load_train_csv(cmd.train_csv);
    std::cout << "[INFO] Train samples: " << train.N << "\n";

    OpticalParams params;
    init_params(params, 1337u);

    std::cout << "[INFO] Training " << cmd.epochs << " epochs...\n";
    train_model(train, params, cmd.epochs, cmd.batch, cmd.lr);

    std::cout << "[INFO] Loading test: " << cmd.test_csv << "\n";
    TestSet test = load_test_csv(cmd.test_csv);
    std::cout << "[INFO] Test samples: " << test.N << "\n";

    std::cout << "[INFO] Running inference...\n";
    auto labels = run_inference(test, params, cmd.batch);

    utils::write_submission_csv(cmd.submission, labels);
    std::cout << "[DONE] Submission written to " << cmd.submission << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FATAL] " << e.what() << "\n";
    return 1;
  }
}
