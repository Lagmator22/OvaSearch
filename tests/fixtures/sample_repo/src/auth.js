const crypto = require('crypto');

/**
 * Hash a password with a random salt using scrypt.
 */
function hashPassword(password) {
  const salt = crypto.randomBytes(16).toString('hex');
  const hash = crypto.scryptSync(password, salt, 64).toString('hex');
  return `${salt}:${hash}`;
}

/**
 * Check a password against a stored salt:hash string.
 */
function verifyPassword(password, stored) {
  const [salt, hash] = stored.split(':');
  const candidate = crypto.scryptSync(password, salt, 64).toString('hex');
  return crypto.timingSafeEqual(Buffer.from(hash, 'hex'), Buffer.from(candidate, 'hex'));
}

/**
 * In-memory session store with expiry.
 */
class SessionStore {
  constructor(ttlMs) {
    this.ttlMs = ttlMs;
    this.sessions = new Map();
  }

  set(id, data) {
    this.sessions.set(id, { data, expires: Date.now() + this.ttlMs });
  }

  get(id) {
    const s = this.sessions.get(id);
    if (!s || s.expires < Date.now()) {
      this.sessions.delete(id);
      return null;
    }
    return s.data;
  }

  destroy(id) {
    this.sessions.delete(id);
  }
}

module.exports = { hashPassword, verifyPassword, SessionStore };
