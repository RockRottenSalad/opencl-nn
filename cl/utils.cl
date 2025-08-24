
ulong rand(ulong x) {
    x ^= (x << 21);
    x ^= (x >> 35);
    x ^= (x << 4);

    return x;
}

// n cannot be larger than the seed buffer or else this explodes
void kernel rand_buffer(
    global float* out,
    const float min,
    const float max,
    const int n,
    global long* seed) {

    const int id = get_global_id(0);
    if(id >= n) return;

    seed[id] = rand(seed[id]);

    float x = (seed[id] / (1L<<63))*(max-min) + min;
    out[id] = x;
}

void kernel zero(global float* out, const uint n) {
    const int id = get_global_id(0);
    if(id != 0) return;

    for(uint i = 0; i < n; i++) out[i] = 0;
}

void kernel copy(global float* dest, global float* src, const uint n) {
    const int id = get_global_id(0);
    if(id >= n) return;

    dest[id] = src[id];
}



