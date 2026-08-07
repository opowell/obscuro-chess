// ---------------------------------------------------------------------------
// A minimal, dependency-free ZIP reader — enough to walk a corpus archive.
//
// A recorded-game corpus arrives as one file far more often than as a directory,
// and the repo's whole shape ("zero dependencies, no build step") rules out
// pulling in a zip library for it. Node's `zlib` is builtin, and the parts of
// the ZIP format a corpus archive actually uses are small: the central
// directory, STORED (method 0) and DEFLATE (method 8), and Zip64 for archives
// that outgrow the 32-bit fields.
//
// RANDOM ACCESS, NOT STREAMING, and that is the point. A corpus zip can be
// gigabytes; `readFileSync` on it would need the whole thing resident before the
// first game is parsed. This reads through a file descriptor: the central
// directory once, then one entry at a time, so peak memory is the largest single
// member rather than the archive. `entries()` is a generator for the same
// reason — the caller can process and discard.
//
// WHAT IS DELIBERATELY NOT HERE: writing, encryption, multi-disk archives, and
// the compression methods nobody uses (bzip2/LZMA/zstd inside zip). Each throws
// a named error rather than being silently skipped, because a corpus that
// half-loads is worse than one that refuses to: the fitter would report a
// confident number on a subset it never mentioned.
// ---------------------------------------------------------------------------

import { closeSync, fstatSync, openSync, readSync } from 'node:fs';
import { inflateRawSync } from 'node:zlib';

const EOCD_SIG = 0x06054b50;        // end of central directory
const EOCD64_SIG = 0x06064b50;      // Zip64 end of central directory
const EOCD64_LOC_SIG = 0x07064b50;  // Zip64 EOCD locator
const CEN_SIG = 0x02014b50;         // central directory file header
const LOC_SIG = 0x04034b50;         // local file header

const STORED = 0, DEFLATED = 8;

/** Read `len` bytes at `pos`. Short reads are a corrupt archive, not a retry. */
function readAt(fd, pos, len) {
  const buf = Buffer.allocUnsafe(len);
  let off = 0;
  while (off < len) {
    const n = readSync(fd, buf, off, len - off, pos + off);
    if (n <= 0) throw new Error(`zip: unexpected end of file at ${pos + off}`);
    off += n;
  }
  return buf;
}

/**
 * Locate the End Of Central Directory record.
 *
 * It is the last thing in the file, but it ends with a variable-length comment,
 * so it has to be found by scanning backwards for its signature. The comment is
 * a 16-bit length, hence the 64 KiB + 22 byte bound: past that, there is no
 * conforming EOCD to find and the file is not a zip.
 */
function findEocd(fd, size) {
  const max = Math.min(size, 0xffff + 22);
  const buf = readAt(fd, size - max, max);
  for (let i = buf.length - 22; i >= 0; i--) {
    if (buf.readUInt32LE(i) !== EOCD_SIG) continue;
    // Guard against the signature appearing inside a file comment: the record's
    // own comment length must account for exactly the bytes that follow it.
    const commentLen = buf.readUInt16LE(i + 20);
    if (i + 22 + commentLen !== buf.length) continue;
    return { at: size - max + i, buf: buf.subarray(i) };
  }
  throw new Error('zip: no end-of-central-directory record — not a zip file');
}

/**
 * Where the central directory starts and how many entries it holds.
 *
 * The 32-bit fields saturate at 0xffff/0xffffffff, at which point the real
 * values live in the Zip64 record that the locator points at. A corpus of a few
 * hundred thousand games hits the entry-count ceiling routinely, so this is not
 * a theoretical branch.
 */
function centralDirectory(fd, size) {
  const { at, buf } = findEocd(fd, size);
  let count = buf.readUInt16LE(10);
  let cdSize = buf.readUInt32LE(12);
  let cdOffset = buf.readUInt32LE(16);

  if (count === 0xffff || cdOffset === 0xffffffff || cdSize === 0xffffffff) {
    if (at < 20) throw new Error('zip: Zip64 fields present but no locator');
    const loc = readAt(fd, at - 20, 20);
    if (loc.readUInt32LE(0) !== EOCD64_LOC_SIG) throw new Error('zip: bad Zip64 locator');
    const eocd64At = Number(loc.readBigUInt64LE(8));
    const e = readAt(fd, eocd64At, 56);
    if (e.readUInt32LE(0) !== EOCD64_SIG) throw new Error('zip: bad Zip64 end-of-central-directory');
    count = Number(e.readBigUInt64LE(32));
    cdSize = Number(e.readBigUInt64LE(40));
    cdOffset = Number(e.readBigUInt64LE(48));
  }
  return { count, cdSize, cdOffset };
}

/**
 * Zip64 extended information (header id 0x0001) in an extra field.
 *
 * The rule is positional, not tagged: the fields appear in a fixed order and
 * ONLY the ones whose 32-bit counterpart was saturated are present. Reading them
 * unconditionally is the classic way to get a garbage offset out of a valid
 * archive, so each read is gated on its own sentinel.
 */
function zip64Extra(extra, need) {
  const out = {};
  let p = 0;
  while (p + 4 <= extra.length) {
    const id = extra.readUInt16LE(p), len = extra.readUInt16LE(p + 2);
    if (id === 0x0001) {
      let q = p + 4;
      if (need.size && q + 8 <= p + 4 + len) { out.size = Number(extra.readBigUInt64LE(q)); q += 8; }
      if (need.csize && q + 8 <= p + 4 + len) { out.csize = Number(extra.readBigUInt64LE(q)); q += 8; }
      if (need.offset && q + 8 <= p + 4 + len) { out.offset = Number(extra.readBigUInt64LE(q)); q += 8; }
      return out;
    }
    p += 4 + len;
  }
  return out;
}

/**
 * Open a zip for reading. Returns `{ entries, read, close }`:
 *
 *   for (const e of z.entries()) if (e.name.endsWith('.pgn')) buf = z.read(e);
 *
 * The handle must be closed. `entries()` may be iterated more than once; the
 * central directory is parsed once, on open.
 */
export function openZip(path) {
  const fd = openSync(path, 'r');
  let members;
  try {
    const size = fstatSync(fd).size;
    const { count, cdOffset } = centralDirectory(fd, size);
    members = [];
    let p = cdOffset;
    for (let i = 0; i < count; i++) {
      const h = readAt(fd, p, 46);
      if (h.readUInt32LE(0) !== CEN_SIG) throw new Error(`zip: bad central header at ${p}`);
      const method = h.readUInt16LE(10);
      let csize = h.readUInt32LE(20);
      let size32 = h.readUInt32LE(24);
      const nameLen = h.readUInt16LE(28);
      const extraLen = h.readUInt16LE(30);
      const commentLen = h.readUInt16LE(32);
      let offset = h.readUInt32LE(42);
      const varBuf = readAt(fd, p + 46, nameLen + extraLen);
      // Bit 11 of the general-purpose flags says the name is UTF-8. Without it
      // the spec says CP437; treating those as UTF-8 mangles only non-ASCII
      // names, and a mangled name still reads back the right bytes, so this
      // does not gate on it.
      const name = varBuf.subarray(0, nameLen).toString('utf8');
      const extra = varBuf.subarray(nameLen);
      const need = { size: size32 === 0xffffffff, csize: csize === 0xffffffff, offset: offset === 0xffffffff };
      if (need.size || need.csize || need.offset) {
        const z = zip64Extra(extra, need);
        if (need.size && z.size != null) size32 = z.size;
        if (need.csize && z.csize != null) csize = z.csize;
        if (need.offset && z.offset != null) offset = z.offset;
      }
      p += 46 + nameLen + extraLen + commentLen;
      // Directory entries carry no data; the trailing slash is the portable
      // test (the external-attributes byte is platform-dependent).
      if (name.endsWith('/')) continue;
      members.push({ name, method, csize, size: size32, offset });
    }
  } catch (err) {
    closeSync(fd);
    throw err;
  }

  return {
    /** Every non-directory member, in central-directory order. */
    *entries() { yield* members; },

    /** One member's bytes, decompressed. */
    read(entry) {
      // The local header repeats the name/extra with its OWN lengths, which are
      // allowed to differ from the central directory's. The data offset must
      // therefore be computed from the local header, never from the central one.
      const lh = readAt(fd, entry.offset, 30);
      if (lh.readUInt32LE(0) !== LOC_SIG) throw new Error(`zip: bad local header for ${entry.name}`);
      const dataAt = entry.offset + 30 + lh.readUInt16LE(26) + lh.readUInt16LE(28);
      const raw = readAt(fd, dataAt, entry.csize);
      if (entry.method === STORED) return raw;
      if (entry.method === DEFLATED) return inflateRawSync(raw);
      throw new Error(`zip: ${entry.name} uses unsupported compression method ${entry.method}`);
    },

    close() { closeSync(fd); },
  };
}
