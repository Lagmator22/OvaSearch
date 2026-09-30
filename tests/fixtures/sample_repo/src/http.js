/**
 * Parse a URL query string like "a=1&b=two" into an object.
 */
const parseQueryString = (qs) => {
  const out = {};
  for (const pair of qs.replace(/^\?/, '').split('&')) {
    if (!pair) continue;
    const [k, v = ''] = pair.split('=');
    out[decodeURIComponent(k)] = decodeURIComponent(v);
  }
  return out;
};

/**
 * Fetch a URL and retry failed requests with exponential backoff.
 */
async function fetchWithRetry(url, retries = 3, delayMs = 200) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(url);
      if (res.ok) return res;
    } catch (err) {
      if (attempt === retries) throw err;
    }
    await new Promise((r) => setTimeout(r, delayMs * 2 ** attempt));
  }
  throw new Error('request failed after retries');
}

module.exports = { parseQueryString, fetchWithRetry };
