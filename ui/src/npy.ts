export interface NpyArray {
  shape: number[];
  data: Float64Array;
}
export function decodeNpy(buffer: ArrayBuffer): NpyArray {
  const bytes = new Uint8Array(buffer),
    view = new DataView(buffer);
  if (
    bytes.length < 12 ||
    bytes[0] !== 0x93 ||
    String.fromCharCode(...bytes.subarray(1, 6)) !== "NUMPY"
  )
    throw new Error("Invalid NumPy file");
  const major = bytes[6];
  if (![1, 2, 3].includes(major))
    throw new Error(`Unsupported NumPy version ${major}`);
  const start = major === 1 ? 10 : 12,
    headerSize =
      major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const header = new TextDecoder().decode(
    bytes.subarray(start, start + headerSize),
  );
  const dtype = /['"]descr['"]\s*:\s*['"]([^'"]+)['"]/.exec(header)?.[1];
  const shape = /['"]shape['"]\s*:\s*\(([^)]*)\)/
    .exec(header)?.[1]
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  if (!dtype || !shape || shape.some((n) => !Number.isSafeInteger(n) || n < 0))
    throw new Error("Invalid NumPy header");
  if (/['"]fortran_order['"]\s*:\s*True/.test(header))
    throw new Error("Fortran-ordered arrays are unsupported");
  const count = shape.reduce((a, b) => a * b, 1),
    width = Number(dtype.slice(2)),
    kind = dtype[1],
    little = dtype[0] !== ">";
  const offset = start + headerSize;
  if (
    !Number.isSafeInteger(count) ||
    count > 100_000_000 ||
    !width ||
    offset + count * width > bytes.length
  )
    throw new Error("Invalid NumPy array size");
  const data = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const p = offset + i * width;
    if (kind === "f" && width === 2)
      data[i] = float16(view.getUint16(p, little));
    else if (kind === "f" && width === 4) data[i] = view.getFloat32(p, little);
    else if (kind === "f" && width === 8) data[i] = view.getFloat64(p, little);
    else if (kind === "i" && width === 8)
      data[i] = Number(view.getBigInt64(p, little));
    else if (kind === "u" && width === 8)
      data[i] = Number(view.getBigUint64(p, little));
    else if (kind === "i" && width === 4) data[i] = view.getInt32(p, little);
    else if (kind === "u" && width === 4) data[i] = view.getUint32(p, little);
    else if (kind === "i" && width === 2) data[i] = view.getInt16(p, little);
    else if (kind === "u" && width === 2) data[i] = view.getUint16(p, little);
    else if (kind === "i" && width === 1) data[i] = view.getInt8(p);
    else if ((kind === "u" || kind === "b") && width === 1)
      data[i] = view.getUint8(p);
    else throw new Error(`Unsupported NumPy dtype ${dtype}`);
  }
  return { shape, data };
}
export function float16(bits: number) {
  const sign = bits & 0x8000 ? -1 : 1,
    exponent = (bits >> 10) & 31,
    fraction = bits & 1023;
  return exponent === 0
    ? sign * 2 ** -14 * (fraction / 1024)
    : exponent === 31
      ? fraction
        ? NaN
        : sign * Infinity
      : sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}
const cache = new Map<string, NpyArray>();
let total = 0;
export async function loadNpy(
  url: string,
  signal?: AbortSignal,
): Promise<NpyArray> {
  const cached = cache.get(url);
  if (cached) {
    cache.delete(url);
    cache.set(url, cached);
    return cached;
  }
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`Array fetch failed: ${response.status}`);
  const array = decodeNpy(await response.arrayBuffer());
  if (array.data.byteLength <= 64 * 1024 * 1024) {
    total -= cache.get(url)?.data.byteLength ?? 0;
    cache.set(url, array);
    total += array.data.byteLength;
    while (total > 64 * 1024 * 1024) {
      const key = cache.keys().next().value!;
      total -= cache.get(key)!.data.byteLength;
      cache.delete(key);
    }
  }
  return array;
}
