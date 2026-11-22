require "./spec_helper"
require "file_utils"
require "../examples/agent_workspace"

# The path guard is the only thing standing between a model-supplied string and the whole
# filesystem, and read_file has no confirmation prompt, so a hole here is silent. Every
# example below is an escape that a plausible model output would attempt.
#
# agent_workspace.cr is required directly rather than through agent.cr: that file has
# top-level code that starts a REPL, so requiring it from a spec would hang the suite. Keeping
# the guard in its own file is what makes it testable at all.
# A fresh root per example, since realpath is resolved against it.
def with_root(&)
  dir = File.tempname("shainet_root")
  Dir.mkdir_p(File.join(dir, "sub"))
  File.write(File.join(dir, "keep.txt"), "hello\n")
  begin
    yield File.realpath(dir)
  ensure
    FileUtils.rm_rf(dir) if Dir.exists?(dir)
  end
end

describe AgentDemo::WorkspacePath do
  it "resolves a plain relative path inside the root" do
    with_root do |root|
      AgentDemo::WorkspacePath.resolve(root, "keep.txt").should eq(File.join(root, "keep.txt"))
    end
  end

  it "resolves a path that does not exist yet, so new files can be created" do
    with_root do |root|
      AgentDemo::WorkspacePath.resolve(root, "sub/new.txt").should eq(File.join(root, "sub", "new.txt"))
    end
  end

  it "refuses an absolute path" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspacePathError, /absolute/) do
        AgentDemo::WorkspacePath.resolve(root, "/etc/passwd")
      end
    end
  end

  it "refuses a traversal out of the root" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspacePathError, /escapes workspace/) do
        AgentDemo::WorkspacePath.resolve(root, "../outside.txt")
      end
    end
  end

  it "refuses a traversal that only escapes after descending" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspacePathError, /escapes workspace/) do
        AgentDemo::WorkspacePath.resolve(root, "sub/../../outside.txt")
      end
    end
  end

  it "refuses an empty path" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspacePathError, /empty/) do
        AgentDemo::WorkspacePath.resolve(root, "   ")
      end
    end
  end

  it "refuses .git at any depth" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspacePathError, /\.git/) do
        AgentDemo::WorkspacePath.resolve(root, ".git/config")
      end
      expect_raises(AgentDemo::WorkspacePathError, /\.git/) do
        AgentDemo::WorkspacePath.resolve(root, "sub/.git/hooks/pre-commit")
      end
    end
  end

  it "refuses a symlink pointing out of the root" do
    with_root do |root|
      outside = File.tempname("shainet_outside")
      Dir.mkdir_p(outside)
      begin
        File.write(File.join(outside, "secret.txt"), "token\n")
        # The classic escape: a link inside the workspace whose target is not.
        File.symlink(outside, File.join(root, "link"))
        expect_raises(AgentDemo::WorkspacePathError, /escapes workspace/) do
          AgentDemo::WorkspacePath.resolve(root, "link/secret.txt")
        end
      ensure
        FileUtils.rm_rf(outside)
      end
    end
  end

  it "refuses a NEW path under a symlinked parent" do
    with_root do |root|
      outside = File.tempname("shainet_outside2")
      Dir.mkdir_p(outside)
      begin
        File.symlink(outside, File.join(root, "link"))
        # The file does not exist, so the check must walk up to the nearest EXISTING ancestor
        # and canonicalize THAT. Checking only the literal path would let this through and a
        # write would land outside the workspace.
        expect_raises(AgentDemo::WorkspacePathError, /escapes workspace/) do
          AgentDemo::WorkspacePath.resolve(root, "link/planted.txt")
        end
      ensure
        FileUtils.rm_rf(outside)
      end
    end
  end

  it "allows the root itself" do
    with_root do |root|
      AgentDemo::WorkspacePath.resolve(root, ".").should eq(root)
    end
  end

  # An out-of-workspace path is NEGOTIABLE, not forbidden, so the escape refusals above raise
  # a distinguishable subclass. The agent asks the user about those and refuses the rest
  # outright; telling the two apart by message text would break on any rewording.
  it "raises the escape subclass for an absolute path" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspaceEscapeError) do
        AgentDemo::WorkspacePath.resolve(root, "/etc/passwd")
      end
    end
  end

  it "raises the escape subclass for a traversal" do
    with_root do |root|
      expect_raises(AgentDemo::WorkspaceEscapeError) do
        AgentDemo::WorkspacePath.resolve(root, "../outside.txt")
      end
    end
  end

  it "does NOT raise the escape subclass for an empty path, which no approval can fix" do
    with_root do |root|
      # A WorkspacePathError that is not a WorkspaceEscapeError: refused outright.
      ex = expect_raises(AgentDemo::WorkspacePathError) do
        AgentDemo::WorkspacePath.resolve(root, "  ")
      end
      ex.class.should eq(AgentDemo::WorkspacePathError)
    end
  end

  it "does NOT raise the escape subclass for a .git path" do
    with_root do |root|
      ex = expect_raises(AgentDemo::WorkspacePathError) do
        AgentDemo::WorkspacePath.resolve(root, ".git/config")
      end
      ex.class.should eq(AgentDemo::WorkspacePathError)
    end
  end
end

describe "AgentDemo::WorkspacePath outside-workspace access" do
  it "resolves an absolute path outside the root" do
    with_root do |root|
      outside = File.tempname("shainet_out")
      Dir.mkdir_p(outside)
      begin
        File.write(File.join(outside, "cfg.json"), "{}\n")
        target = File.join(File.realpath(outside), "cfg.json")
        AgentDemo::WorkspacePath.resolve_anywhere(root, target).should eq(target)
      ensure
        FileUtils.rm_rf(outside)
      end
    end
  end

  it "canonicalizes an approved outside path through a symlink" do
    with_root do |root|
      real = File.tempname("shainet_real")
      Dir.mkdir_p(real)
      link = File.tempname("shainet_link")
      begin
        File.write(File.join(real, "f.txt"), "x\n")
        File.symlink(real, link)
        # Approving a path must record the REAL location: without realpath the approval would
        # name a label that could later point somewhere else.
        AgentDemo::WorkspacePath.resolve_anywhere(root, File.join(link, "f.txt"))
          .should eq(File.join(File.realpath(real), "f.txt"))
      ensure
        File.delete(link) if File.symlink?(link)
        FileUtils.rm_rf(real)
      end
    end
  end

  it "still refuses .git outside the workspace" do
    with_root do |root|
      outside = File.tempname("shainet_outgit")
      Dir.mkdir_p(File.join(outside, ".git"))
      begin
        expect_raises(AgentDemo::WorkspacePathError, /\.git/) do
          AgentDemo::WorkspacePath.resolve_anywhere(root, File.join(File.realpath(outside), ".git", "config"))
        end
      ensure
        FileUtils.rm_rf(outside)
      end
    end
  end

  it "remembers an approval per path, so the same file is not asked about twice" do
    AgentDemo::WorkspacePath.reset_outside_approvals!
    begin
      AgentDemo::WorkspacePath.approved_outside?("/tmp/a.txt").should be_false
      AgentDemo::WorkspacePath.approve_outside("/tmp/a.txt")
      # The whole claim of "one time approval": asked once, remembered after.
      AgentDemo::WorkspacePath.approved_outside?("/tmp/a.txt").should be_true
    ensure
      AgentDemo::WorkspacePath.reset_outside_approvals!
    end
  end

  it "does not let one approval cover a DIFFERENT outside path" do
    AgentDemo::WorkspacePath.reset_outside_approvals!
    begin
      AgentDemo::WorkspacePath.approve_outside("/home/u/models/cfg.json")
      # The reason approval is per path: approving a model config must not also grant this.
      AgentDemo::WorkspacePath.approved_outside?("/home/u/.ssh/id_rsa").should be_false
    ensure
      AgentDemo::WorkspacePath.reset_outside_approvals!
    end
  end

  it "forgets every approval on reset, so /ask fully revokes" do
    AgentDemo::WorkspacePath.reset_outside_approvals!
    AgentDemo::WorkspacePath.approve_outside("/tmp/x")
    AgentDemo::WorkspacePath.approve_outside("/tmp/y")
    AgentDemo::WorkspacePath.approved_outside_count.should eq(2)
    AgentDemo::WorkspacePath.reset_outside_approvals!
    AgentDemo::WorkspacePath.approved_outside_count.should eq(0)
    AgentDemo::WorkspacePath.approved_outside?("/tmp/x").should be_false
  end
end

# /workspace <dir> resolves what the user typed. It once raised File::NotFoundError for a mistyped
# path, which went uncaught and ended the whole agent session.
describe "AgentDemo::WorkspacePath.resolve_root" do
  it "expands a leading ~ to the home directory" do
    AgentDemo::WorkspacePath.resolve_root("~").should eq(File.realpath(Path.home.to_s))
    with_root do |root|
      rel = root.lchop(File.realpath(Path.home.to_s))
      next unless rel != root # tempdir not under $HOME on this machine
      AgentDemo::WorkspacePath.resolve_root("~#{rel}").should eq(root)
    end
  end

  it "raises ArgumentError, not File::Error, for a path that does not exist" do
    expect_raises(ArgumentError, /no such directory/) do
      AgentDemo::WorkspacePath.resolve_root("~/.kiro/crew/scra~/.kiro/crew/scratch/nope/")
    end
  end

  it "raises ArgumentError for a file" do
    with_root do |root|
      expect_raises(ArgumentError, /not a directory/) do
        AgentDemo::WorkspacePath.resolve_root(File.join(root, "keep.txt"))
      end
    end
  end

  it "canonicalizes an existing directory" do
    with_root do |root|
      AgentDemo::WorkspacePath.resolve_root(File.join(root, "sub", "..")).should eq(root)
    end
  end
end
