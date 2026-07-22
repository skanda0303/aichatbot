/**
 * audioPlayer.js — Ordered audio clip queue and playback for the RAG chatbot.
 *
 * Receives base64-encoded audio clips tagged with a monotonically increasing
 * index. Plays them in strict ascending order regardless of arrival order.
 * Handles browser autoplay restrictions by tying playback to user interaction
 * (send-button click sets a flag that unlocks the first play).
 *
 * Usage:
 *   window.audioPlayer.enqueue(index, base64String)
 *   window.audioPlayer.reset()
 *   window.audioPlayer.setEnabled(true/false)
 *   window.audioPlayer.userInteracted()   // call on send-button click
 */

class AudioQueuePlayer {
  constructor() {
    /** @type {Map<number, string>} index → base64 audio string */
    this.queue = new Map();
    this.nextIndex = 0;
    this.currentAudio = null;
    this.enabled = false;
    this._userInteracted = false;
    this._playing = false;
  }

  /**
   * Signal that the user has just interacted (clicked Send).
   * Required to satisfy browser autoplay policies.
   */
  userInteracted() {
    this._userInteracted = true;
  }

  /**
   * Add an audio clip to the queue.
   * @param {number} index   - Sequential index of this sentence clip.
   * @param {string} b64     - Base64-encoded audio bytes (mp3).
   */
  enqueue(index, b64) {
    this.queue.set(index, b64);
    this._tryPlayNext();
  }

  /**
   * Attempt to play the next clip in sequence.
   * Does nothing if already playing, disabled, or the next index hasn't arrived yet.
   */
  _tryPlayNext() {
    if (!this.enabled || !this._userInteracted || this._playing) return;
    if (!this.queue.has(this.nextIndex)) return;

    const b64 = this.queue.get(this.nextIndex);
    this.queue.delete(this.nextIndex);
    this.nextIndex++;
    this._playing = true;

    // Decode base64 → Uint8Array → Blob → Object URL
    try {
      const binaryStr = atob(b64);
      const bytes = new Uint8Array(binaryStr.length);
      for (let i = 0; i < binaryStr.length; i++) {
        bytes[i] = binaryStr.charCodeAt(i);
      }
      const blob = new Blob([bytes], { type: "audio/mpeg" });
      const url = URL.createObjectURL(blob);

      const audio = new Audio(url);
      this.currentAudio = audio;

      // Show playing indicator
      this._showIndicator(true);

      audio.onended = () => {
        URL.revokeObjectURL(url);
        this._playing = false;
        this.currentAudio = null;
        this._tryPlayNext();   // advance to next clip
      };

      audio.onerror = () => {
        console.warn("[AudioPlayer] Playback error on clip", this.nextIndex - 1);
        URL.revokeObjectURL(url);
        this._playing = false;
        this.currentAudio = null;
        this._tryPlayNext();
      };

      audio.play().catch((err) => {
        // Autoplay blocked despite user-interaction flag — degrade silently
        console.warn("[AudioPlayer] play() rejected:", err);
        URL.revokeObjectURL(url);
        this._playing = false;
        this.currentAudio = null;
        this._showIndicator(false);
      });

    } catch (err) {
      console.error("[AudioPlayer] Decode error:", err);
      this._playing = false;
      this._showIndicator(false);
    }
  }

  /**
   * Stop any current playback and clear all queued clips.
   * Call this on /clear and on new message submission.
   */
  reset() {
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.src = "";
      this.currentAudio = null;
    }
    this.queue.clear();
    this.nextIndex = 0;
    this._playing = false;
    this._userInteracted = false;
    this._showIndicator(false);
  }

  /**
   * Enable or disable the player.
   * If disabled while playing, stops immediately.
   * @param {boolean} flag
   */
  setEnabled(flag) {
    this.enabled = flag;
    if (!flag) {
      this.reset();
    }
  }

  /**
   * Show or hide the playing indicator element (#voice-playing-indicator).
   * @param {boolean} visible
   */
  _showIndicator(visible) {
    const el = document.getElementById("voice-playing-indicator");
    if (!el) return;
    if (visible) {
      el.style.display = "inline-flex";
    } else {
      el.style.display = "none";
      this._checkAllDone();
    }
  }

  /**
   * Hide indicator if queue is empty and not playing.
   */
  _checkAllDone() {
    if (!this._playing && this.queue.size === 0) {
      const el = document.getElementById("voice-playing-indicator");
      if (el) el.style.display = "none";
    }
  }
}

// Expose singleton globally
window.audioPlayer = new AudioQueuePlayer();
