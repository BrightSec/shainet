# Workspace path confinement for the agent's file tools.
#
# Lifted from bar-bot (src/product/workspace_path.cr, MIT, same author) so the two agents
# share one guard rather than growing two subtly different ones.
#
# Every file tool resolves through this. Without it each tool took a raw path, and read_file
# has no confirmation gate, so a model could read ~/.aws/credentials or ~/.ssh/id_rsa
# unprompted and write anywhere in $HOME on a single "y".
#
# The rules that matter:
#   * absolute paths are refused outright, so the model must stay relative to the root
#   * realpath canonicalization, so a symlink cannot point out of the workspace
#   * for a path that does not exist yet, the nearest EXISTING ancestor is canonicalized and
#     re-checked -- otherwise a new file under a symlinked parent would escape
#   * .git is refused, since a tool rewriting git internals is never what was asked for
module AgentDemo
  class WorkspacePathError < Exception
  end

  # A path that is well-formed but lies OUTSIDE the workspace root, either because it is
  # absolute or because it traverses out.
  #
  # Separate from its parent so a caller can tell a NEGOTIABLE refusal from a flat one: an
  # outside path can be allowed by asking the user, whereas an empty path or a .git path is
  # wrong no matter who approves it. Distinguishing those by matching on message text would
  # break the moment a message is reworded.
  class WorkspaceEscapeError < WorkspacePathError
  end

  module WorkspacePath
    # A user-typed directory for /workspace, canonicalized. No shell is involved, so a leading ~ is
    # expanded here (File.realpath alone looks for a directory named "~"). Raises ArgumentError,
    # never File::Error, so the REPL reports a bad path instead of dying on it.
    def self.resolve_root(path : String) : String
      expanded = path == "~" || path.starts_with?("~/") ? File.join(Path.home.to_s, path.lchop("~")) : path
      resolved = begin
        File.realpath(expanded)
      rescue File::Error
        raise ArgumentError.new("no such directory: #{path}")
      end
      raise ArgumentError.new("not a directory: #{path}") unless Dir.exists?(resolved)
      resolved
    end

    extend self

    def resolve(root : String, relative : String) : String
      requested = relative.strip
      raise WorkspacePathError.new("path is empty") if requested.empty?
      raise WorkspaceEscapeError.new("absolute paths are not allowed") if Path[requested].absolute?

      canonical_root = File.realpath(root)
      candidate = File.expand_path(requested, canonical_root)
      ensure_within_root!(canonical_root, candidate)
      reject_git_path!(canonical_root, candidate)

      existing = candidate
      until File.exists?(existing) || Dir.exists?(existing)
        parent = File.dirname(existing)
        raise WorkspaceEscapeError.new("path escapes workspace") if parent == existing
        existing = parent
      end

      canonical_existing = File.realpath(existing)
      ensure_within_root!(canonical_root, canonical_existing)
      candidate
    rescue ex : WorkspacePathError
      raise ex
    rescue ex
      raise WorkspacePathError.new("invalid workspace path: #{ex.message}")
    end

    # Resolve a path WITHOUT confining it to the root, for a location the user has explicitly
    # approved. Everything except the confinement still applies.
    #
    # This exists so an approved out-of-workspace path is still canonicalized rather than used
    # raw: the point of approving `~/models/config.json` is that one file, and without
    # realpath a symlink there could redirect the write somewhere else entirely. The caller
    # owns the approval decision; this only makes the path safe to act on once approved.
    def resolve_anywhere(root : String, requested : String) : String
      path = requested.strip
      raise WorkspacePathError.new("path is empty") if path.empty?
      candidate = Path[path].absolute? ? File.expand_path(path) : File.expand_path(path, File.realpath(root))

      existing = candidate
      until File.exists?(existing) || Dir.exists?(existing)
        parent = File.dirname(existing)
        break if parent == existing
        existing = parent
      end
      # Canonicalize through the nearest existing ancestor, so the approval covers the real
      # location and not a symlink's label.
      canonical = File.exists?(candidate) || Dir.exists?(candidate) ? File.realpath(candidate) : begin
        base = File.realpath(existing)
        rest = candidate[existing.size..].lstrip('/')
        rest.empty? ? base : File.join(base, rest)
      end
      # .git stays refused even here: rewriting git internals is never the task, and an
      # approval for a directory should not silently include its repository plumbing.
      if canonical.split('/').includes?(".git")
        raise WorkspacePathError.new(".git paths are not editable by the agent")
      end
      canonical
    rescue ex : WorkspacePathError
      raise ex
    rescue ex
      raise WorkspacePathError.new("invalid path: #{ex.message}")
    end

    # Out-of-workspace locations the user has approved this session.
    #
    # Approval is remembered PER PATH rather than as one global yes. Being asked again for a
    # file already approved is the annoyance worth removing; having one approval of a model
    # config silently also cover ~/.ssh/id_rsa is not. Process-scoped and never persisted,
    # like the allow-all flag.
    @@approved_outside = Set(String).new

    def self.approved_outside?(path : String) : Bool
      @@approved_outside.includes?(path)
    end

    def self.approve_outside(path : String) : Nil
      @@approved_outside << path
    end

    def self.reset_outside_approvals! : Nil
      @@approved_outside.clear
    end

    def self.approved_outside_count : Int32
      @@approved_outside.size
    end

    private def ensure_within_root!(root : String, candidate : String)
      return if candidate == root || candidate.starts_with?("#{root}/")
      raise WorkspaceEscapeError.new("path escapes workspace")
    end

    private def reject_git_path!(root : String, candidate : String)
      relative = candidate == root ? "" : candidate[(root.size + 1)..]
      if relative.split('/').includes?(".git")
        raise WorkspacePathError.new(".git paths are managed by the worker")
      end
    end
  end
end
