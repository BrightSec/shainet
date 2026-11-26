# Draft acceptance for MTP speculative decoding in the OpenAI server.
#
# Its own file because openai_server.cr starts the HTTP server at top level, so a spec cannot require
# it; this is the part of the speculative loop whose order of operations decides what the client
# receives, so it is the part that needs a spec.
module OpenAIServer
  # Returned by the row sampler for a row whose sampled logit is not finite.
  NONFINITE = -2

  enum DraftOutcome
    # Every kept token was taken; the token after them is ready for the next step.
    Continue
    # `accept` asked to stop (a stop token or string, or the limit).
    Stopped
    # A row's logit was not finite; generation ends as one-at-a-time decoding would.
    Nonfinite
  end

  # Walk a verify pass's rows. Row i is sampled with `sample_row.call(i)`; while it equals draft i the
  # draft is kept and passed to `accept`, in order, so penalties and stop checks see each kept token
  # before the next row is sampled. The first row that disagrees gives the next token instead.
  #
  # Returns {drafts kept, next token (-1 unless Continue), outcome}.
  def self.accept_drafts(drafts : Array(Int32), sample_row : Proc(Int32, Int32),
                         accept : Proc(Int32, Bool)) : {Int32, Int32, DraftOutcome}
    a = 0
    loop do
      s = sample_row.call(a)
      return {a, -1, DraftOutcome::Nonfinite} if s == NONFINITE
      return {a, s, DraftOutcome::Continue} unless a < drafts.size && s == drafts[a]
      a += 1
      return {a, -1, DraftOutcome::Stopped} unless accept.call(s)
    end
  end
end
