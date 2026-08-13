#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <thread>

#include "broker.hpp"

namespace fs = std::filesystem;
using namespace snapshot::pagebroker;

namespace {

class BrokerTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    root_ = fs::temp_directory_path() / "pagebroker-daemon-tests" /
            ::testing::UnitTest::GetInstance()->current_test_info()->name();
    fs::remove_all(root_);
    source_ = root_ / "storage" / "source";
    fs::create_directories(source_);
    std::ofstream(source_ / "image") << "image";
    broker_.emplace(root_ / "tmpfs");
  }

  void TearDown() override { fs::remove_all(root_); }

  Request RequestFor(const std::string& id)
  {
    Request request;
    request.set_request_id("request-" + id + "-" + std::to_string(++request_number_));
    request.set_transaction_id(id);
    return request;
  }

  void Configure(StorageBackend* storage, IOEngine* engine, const fs::path& directory)
  {
    storage->mutable_filesystem()->set_directory(directory.string());
    engine->mutable_posix_copy();
  }

  Broker& broker() { return *broker_; }

  fs::path root_;
  fs::path source_;
  std::optional<Broker> broker_;
  unsigned request_number_ = 0;
};

TEST_F(BrokerTest, StagesRestoreAndCleansUpOnCommit)
{
  auto restore = RequestFor("restore");
  Configure(
      restore.mutable_staged_restore()->mutable_source(), restore.mutable_staged_restore()->mutable_io_engine(),
      source_);
  const auto staged = broker().HandleRequest(restore);
  ASSERT_TRUE(staged.has_staged_restore_directory());
  const fs::path staging_directory(staged.staged_restore_directory().image_directory());
  EXPECT_TRUE(fs::exists(staging_directory / "image"));

  const auto conflict = broker().HandleRequest(restore);
  ASSERT_TRUE(conflict.has_failure());
  EXPECT_EQ(conflict.failure().code(), Failure::TRANSACTION_CONFLICT);

  auto commit = RequestFor("restore");
  commit.mutable_commit();
  EXPECT_TRUE(broker().HandleRequest(commit).has_commit_complete());
  EXPECT_TRUE(broker().HandleRequest(commit).has_commit_complete());
  EXPECT_FALSE(fs::exists(staging_directory));

  auto abort = RequestFor("restore");
  abort.mutable_abort();
  const auto abort_response = broker().HandleRequest(abort);
  ASSERT_TRUE(abort_response.has_failure());
  EXPECT_EQ(abort_response.failure().code(), Failure::TRANSACTION_NOT_FOUND);
}

TEST_F(BrokerTest, StagesIndependentRestoresConcurrently)
{
  auto first = RequestFor("first");
  auto second = RequestFor("second");
  Configure(
      first.mutable_staged_restore()->mutable_source(), first.mutable_staged_restore()->mutable_io_engine(), source_);
  Configure(
      second.mutable_staged_restore()->mutable_source(), second.mutable_staged_restore()->mutable_io_engine(), source_);

  Response first_response;
  Response second_response;
  std::thread first_request([&] { first_response = broker().HandleRequest(first); });
  std::thread second_request([&] { second_response = broker().HandleRequest(second); });
  first_request.join();
  second_request.join();

  ASSERT_TRUE(first_response.has_staged_restore_directory());
  ASSERT_TRUE(second_response.has_staged_restore_directory());
  EXPECT_NE(
      first_response.staged_restore_directory().image_directory(),
      second_response.staged_restore_directory().image_directory());
}

TEST_F(BrokerTest, RejectsConcurrentRestoreForSameTransaction)
{
  auto first = RequestFor("restore");
  auto second = RequestFor("restore");
  Configure(
      first.mutable_staged_restore()->mutable_source(), first.mutable_staged_restore()->mutable_io_engine(), source_);
  Configure(
      second.mutable_staged_restore()->mutable_source(), second.mutable_staged_restore()->mutable_io_engine(), source_);

  Response first_response;
  Response second_response;
  std::thread first_request([&] { first_response = broker().HandleRequest(first); });
  std::thread second_request([&] { second_response = broker().HandleRequest(second); });
  first_request.join();
  second_request.join();

  EXPECT_NE(first_response.has_staged_restore_directory(), second_response.has_staged_restore_directory());
}

TEST_F(BrokerTest, RejectsUnsafeTransactionIDs)
{
  for (const auto& id :
       {std::string("../escape"), std::string("nested/name"), std::string("nested\\name"),
        std::string("nul\0suffix", 10)}) {
    auto abort = RequestFor(id);
    abort.mutable_abort();
    const auto response = broker().HandleRequest(abort);
    EXPECT_TRUE(response.has_failure());
    EXPECT_EQ(response.failure().code(), Failure::INVALID_REQUEST);
  }
}

TEST_F(BrokerTest, RejectsSymlinkInRestoreSource)
{
  fs::create_symlink(root_ / "storage" / "elsewhere", source_ / "link");
  auto restore = RequestFor("symlink");
  Configure(
      restore.mutable_staged_restore()->mutable_source(), restore.mutable_staged_restore()->mutable_io_engine(),
      source_);
  const auto response = broker().HandleRequest(restore);
  ASSERT_TRUE(response.has_failure());
  EXPECT_EQ(response.failure().code(), Failure::STORAGE_ERROR);
}

TEST_F(BrokerTest, InvalidRestoreDoesNotReserveTransaction)
{
  auto invalid = RequestFor("restore");
  Configure(
      invalid.mutable_staged_restore()->mutable_source(), invalid.mutable_staged_restore()->mutable_io_engine(),
      "relative");
  EXPECT_EQ(broker().HandleRequest(invalid).failure().code(), Failure::INVALID_REQUEST);

  auto restore = RequestFor("restore");
  Configure(
      restore.mutable_staged_restore()->mutable_source(), restore.mutable_staged_restore()->mutable_io_engine(),
      source_);
  EXPECT_TRUE(broker().HandleRequest(restore).has_staged_restore_directory());
}

TEST_F(BrokerTest, RejectsInvalidStagedRestore)
{
  auto restore = RequestFor("restore");
  restore.mutable_staged_restore();
  const auto response = broker().HandleRequest(restore);
  ASSERT_TRUE(response.has_failure());
  EXPECT_EQ(response.failure().code(), Failure::INVALID_REQUEST);
}

TEST_F(BrokerTest, UnknownCommitAndAbortDoNotReserveTransactions)
{
  auto commit = RequestFor("unknown-commit");
  commit.mutable_commit();
  EXPECT_EQ(broker().HandleRequest(commit).failure().code(), Failure::TRANSACTION_NOT_FOUND);

  auto restore = RequestFor("unknown-commit");
  Configure(
      restore.mutable_staged_restore()->mutable_source(), restore.mutable_staged_restore()->mutable_io_engine(),
      source_);
  EXPECT_TRUE(broker().HandleRequest(restore).has_staged_restore_directory());

  auto abort = RequestFor("unknown-abort");
  abort.mutable_abort();
  EXPECT_EQ(broker().HandleRequest(abort).failure().code(), Failure::TRANSACTION_NOT_FOUND);

  auto prepare = RequestFor("unknown-abort");
  Configure(
      prepare.mutable_prepare_staged_checkpoint()->mutable_destination(),
      prepare.mutable_prepare_staged_checkpoint()->mutable_io_engine(), root_ / "storage" / "published");
  EXPECT_TRUE(broker().HandleRequest(prepare).has_staged_checkpoint_directory());
}

TEST_F(BrokerTest, EvictsOldTerminalTransactionsButRetainsRecentCompletions)
{
  auto oldest = RequestFor("oldest");
  Configure(
      oldest.mutable_prepare_staged_checkpoint()->mutable_destination(),
      oldest.mutable_prepare_staged_checkpoint()->mutable_io_engine(), root_ / "storage" / "oldest");
  ASSERT_TRUE(broker().HandleRequest(oldest).has_staged_checkpoint_directory());
  auto oldest_commit = RequestFor("oldest");
  oldest_commit.mutable_commit();
  ASSERT_TRUE(broker().HandleRequest(oldest_commit).has_commit_complete());

  for (size_t index = 0; index < 1024; ++index) {
    const auto id = "terminal-" + std::to_string(index);
    auto prepare = RequestFor(id);
    Configure(
        prepare.mutable_prepare_staged_checkpoint()->mutable_destination(),
        prepare.mutable_prepare_staged_checkpoint()->mutable_io_engine(), root_ / "storage" / id);
    ASSERT_TRUE(broker().HandleRequest(prepare).has_staged_checkpoint_directory());
    auto commit = RequestFor(id);
    commit.mutable_commit();
    ASSERT_TRUE(broker().HandleRequest(commit).has_commit_complete());
  }

  auto recent_commit = RequestFor("terminal-1023");
  recent_commit.mutable_commit();
  EXPECT_TRUE(broker().HandleRequest(recent_commit).has_commit_complete());

  auto reuse = RequestFor("oldest");
  Configure(
      reuse.mutable_staged_restore()->mutable_source(), reuse.mutable_staged_restore()->mutable_io_engine(), source_);
  EXPECT_TRUE(broker().HandleRequest(reuse).has_staged_restore_directory());
}

TEST_F(BrokerTest, AbortsFailedCheckpointStaging)
{
  const fs::path destination = root_ / "storage" / "published";
  const fs::path staging_directory = root_ / "tmpfs" / "checkpoint" / "checkpoint";
  fs::create_symlink(root_ / "missing", staging_directory);
  auto prepare = RequestFor("checkpoint");
  Configure(
      prepare.mutable_prepare_staged_checkpoint()->mutable_destination(),
      prepare.mutable_prepare_staged_checkpoint()->mutable_io_engine(), destination);
  const auto failed = broker().HandleRequest(prepare);
  ASSERT_TRUE(failed.has_failure());
  EXPECT_EQ(failed.failure().code(), Failure::STORAGE_ERROR);

  auto abort = RequestFor("checkpoint");
  abort.mutable_abort();
  EXPECT_TRUE(broker().HandleRequest(abort).has_abort_complete());

  const auto retry = broker().HandleRequest(prepare);
  ASSERT_TRUE(retry.has_failure());
  EXPECT_EQ(retry.failure().code(), Failure::TRANSACTION_CONFLICT);
}

TEST_F(BrokerTest, PublishesCheckpoint)
{
  const fs::path published = root_ / "storage" / "published";
  auto prepare = RequestFor("checkpoint");
  Configure(
      prepare.mutable_prepare_staged_checkpoint()->mutable_destination(),
      prepare.mutable_prepare_staged_checkpoint()->mutable_io_engine(), published);
  const auto output = broker().HandleRequest(prepare);
  ASSERT_TRUE(output.has_staged_checkpoint_directory());
  const fs::path staging_directory(output.staged_checkpoint_directory().image_directory());
  std::ofstream(staging_directory / "image") << "image";
  std::ofstream(staging_directory / ".destination") << "image";

  auto commit = RequestFor("checkpoint");
  commit.mutable_commit();
  EXPECT_TRUE(broker().HandleRequest(commit).has_commit_complete());
  EXPECT_TRUE(fs::exists(published / "image"));
  EXPECT_TRUE(fs::exists(published / ".destination"));
  EXPECT_FALSE(fs::exists(staging_directory));
}

TEST_F(BrokerTest, ReplacesExistingCheckpoint)
{
  const fs::path published = root_ / "storage" / "published";
  fs::create_directories(published);
  std::ofstream(published / "old") << "old";

  auto prepare = RequestFor("checkpoint");
  Configure(
      prepare.mutable_prepare_staged_checkpoint()->mutable_destination(),
      prepare.mutable_prepare_staged_checkpoint()->mutable_io_engine(), published);
  const auto output = broker().HandleRequest(prepare);
  ASSERT_TRUE(output.has_staged_checkpoint_directory());
  std::ofstream(fs::path(output.staged_checkpoint_directory().image_directory()) / "new") << "new";

  auto commit = RequestFor("checkpoint");
  commit.mutable_commit();
  EXPECT_TRUE(broker().HandleRequest(commit).has_commit_complete());
  EXPECT_FALSE(fs::exists(published / "old"));
  EXPECT_TRUE(fs::exists(published / "new"));
}

TEST_F(BrokerTest, PreservesExistingPartialCheckpointDestination)
{
  const fs::path destination = root_ / "storage" / "blocked";
  auto prepare = RequestFor("checkpoint");
  Configure(
      prepare.mutable_prepare_staged_checkpoint()->mutable_destination(),
      prepare.mutable_prepare_staged_checkpoint()->mutable_io_engine(), destination);
  const auto output = broker().HandleRequest(prepare);
  ASSERT_TRUE(output.has_staged_checkpoint_directory());
  std::ofstream(fs::path(output.staged_checkpoint_directory().image_directory()) / "image") << "image";
  const fs::path partial = destination.string() + ".pagebroker-partial";
  std::ofstream(partial) << "keep";

  auto commit = RequestFor("checkpoint");
  commit.mutable_commit();
  const auto response = broker().HandleRequest(commit);
  ASSERT_TRUE(response.has_failure());
  EXPECT_EQ(response.failure().code(), Failure::TRANSACTION_CONFLICT);
  EXPECT_TRUE(fs::exists(partial));
  std::string partial_contents;
  std::ifstream(partial) >> partial_contents;
  EXPECT_EQ(partial_contents, "keep");
}

TEST_F(BrokerTest, AbortsRestore)
{
  auto restore = RequestFor("restore");
  Configure(
      restore.mutable_staged_restore()->mutable_source(), restore.mutable_staged_restore()->mutable_io_engine(),
      source_);
  const auto staged = broker().HandleRequest(restore);
  ASSERT_TRUE(staged.has_staged_restore_directory());
  const fs::path staging_directory(staged.staged_restore_directory().image_directory());

  auto abort = RequestFor("restore");
  abort.mutable_abort();
  EXPECT_TRUE(broker().HandleRequest(abort).has_abort_complete());
  EXPECT_TRUE(broker().HandleRequest(abort).has_abort_complete());
  EXPECT_FALSE(fs::exists(staging_directory));

  auto commit = RequestFor("restore");
  commit.mutable_commit();
  const auto commit_response = broker().HandleRequest(commit);
  ASSERT_TRUE(commit_response.has_failure());
  EXPECT_EQ(commit_response.failure().code(), Failure::TRANSACTION_NOT_FOUND);
}

}  // namespace
