//
//  pickle.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

/// \file
/// \brief A restricted unpickler: the opcodes and globals of a PyTorch state dict, and nothing else.
/// \ingroup pytorch
///
/// \overview
/// A pickle is a program for a stack machine, and a general unpickler runs it: `GLOBAL os.system` and `REDUCE` is
/// arbitrary code. This one interprets only what `torch.save(state_dict)` emits and builds plain data from it, never
/// calling anything:
///
/// - Opcodes: PROTO, FRAME, STOP, MARK, the memo (PUT, BINPUT, LONG_BINPUT, GET, BINGET, LONG_BINGET, MEMOIZE), NONE,
///   NEWTRUE, NEWFALSE, the integers (BININT, BININT1, BININT2, LONG1 up to 8 bytes), BINFLOAT, the strings
///   (BINUNICODE, SHORT_BINUNICODE, BINUNICODE8), the bytes (BINBYTES, SHORT_BINBYTES), the containers (EMPTY_TUPLE,
///   TUPLE, TUPLE1 to TUPLE3, EMPTY_LIST, APPEND, APPENDS, EMPTY_DICT, SETITEM, SETITEMS), GLOBAL, STACK_GLOBAL,
///   REDUCE, BUILD, BINPERSID.
/// - Globals: `collections.OrderedDict`, `torch._utils._rebuild_tensor_v2`, `torch._utils._rebuild_parameter`, and
///   the storage types (`torch.FloatStorage` ... `torch.BoolStorage`, the bfloat16 and float8 ones).
/// - REDUCE of those globals only: `OrderedDict()` is a dict; `_rebuild_tensor_v2(storage, offset, size, stride, ...)`
///   is a tensor record; `_rebuild_parameter(tensor, ...)` is that tensor. BUILD is accepted on a dict or tensor and
///   its state discarded (an OrderedDict's `_metadata`). BINPERSID takes PyTorch's
///   `('storage', type, key, location, numel)` and makes a storage record.
///
/// Every other opcode and every other global is `unsafe_pickle`, naming it. Limits keep a hostile pickle from
/// exhausting memory: objects, memo entries and nesting are bounded, and every length is checked against the bytes.
///
/// \relations
/// \see formats::pytorch::parse (the zip container around it)
///
/// \impls
/// One pass over the opcodes into an arena of objects referenced by index (so the memo and in-place SETITEMS behave
/// as Python's do).
///
/// \contracts
/// Nothing in the pickle is executed. `unpickle` never reads outside its span.
///
/// \errors
/// `unsafe_pickle` (an opcode or global outside the set, a REDUCE of anything else), `truncated`, `bad_header` (a stack
/// underflow, a bad memo index, a malformed persistent id), `too_large` (a limit).
///
/// \examples
/// \snippet pickle_demo.cpp pickle
///
/// \perf
/// Not benchmarked: linear in the pickle, which holds no tensor data.
///
/// \testing
/// tests/pytorch/pickle_test.cpp (hand-assembled pickles, including `os.system` and `builtins.eval` attempts),
/// demos/pytorch/pickle_demo.cpp.
///
/// \deps
/// The standard library and core.
///
/// \gotchas
/// Pickles of protocol 2 to 5 are read; protocol 0 and 1 text opcodes (INT, STRING, ...) are refused.
///
/// \status
/// Experimental.

#ifndef snicholls_formats_pytorch_pickle_hpp
#define snicholls_formats_pytorch_pickle_hpp

#include <snicholls/formats/error.hpp>

#include <snicholls/core/endian.hpp>

#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace snicholls::formats::pytorch
{
/// \brief What an unpickled object is.
enum class kind : std::uint8_t
{
    none,       ///< None.
    boolean,    ///< True or False (`integer` holds 0 or 1).
    integer,    ///< An int that fits int64.
    floating,   ///< A float.
    string,     ///< A str (`text`).
    bytes,      ///< A bytes (`text`, raw).
    tuple,      ///< A tuple (`items`).
    list,       ///< A list (`items`).
    dict,       ///< A dict or OrderedDict (`items`: key, value, key, value, ...).
    global,     ///< An allowed global (`text`: "module.name").
    storage,    ///< A storage record: `text` its type ("FloatStorage"), `key` its archive key, `integer` its elements.
    tensor,     ///< A tensor record: `items` = storage, offset, size tuple, stride tuple.
    mark,       ///< The MARK on the stack (never in a result).
};

/// \brief One object; containers refer to others by index.
struct object
{
    pytorch::kind kind = pytorch::kind::none;   ///< What it is.
    std::int64_t integer = 0;                   ///< An int, a bool, a storage's element count.
    double floating = 0;                        ///< A float.
    std::string text;                           ///< A string, bytes, a global's name, a storage's type.
    std::string key;                            ///< A storage's key in the archive.
    std::vector<std::size_t> items;             ///< A container's elements (indices into the arena).
};

/// \brief An unpickled object graph.
struct pickled
{
    std::vector<object> objects;   ///< Every object made.
    std::size_t root = 0;          ///< The object STOP returned.

    /// \brief An object by index. \param i The index. \return The object.
    [[nodiscard]] const object& at(std::size_t i) const { return objects.at(i); }
};

/// \brief The limits `unpickle` enforces.
struct unpickle_limits
{
    std::size_t max_objects = 1u << 24;   ///< Objects in all.
    std::size_t max_memo = 1u << 24;      ///< Memo entries.
    std::size_t max_stack = 1u << 20;     ///< The stack's depth.
};

/// \brief Whether a global may be named. \param module The module. \param name The name. \return True for the set
/// above.
[[nodiscard]] constexpr bool allowed_global(std::string_view module, std::string_view name) noexcept
{
    if (module == "collections")
        return name == "OrderedDict";
    if (module == "torch._utils")
        return name == "_rebuild_tensor_v2" || name == "_rebuild_parameter";
    if (module == "torch") {
        constexpr std::string_view storages[] = {"DoubleStorage", "FloatStorage",  "HalfStorage", "BFloat16Storage", "LongStorage",
                                                 "IntStorage",    "ShortStorage",  "CharStorage", "ByteStorage",     "BoolStorage",
                                                 "Float8_e4m3fnStorage", "Float8_e5m2Storage"};
        for (const auto s : storages)
            if (s == name)
                return true;
    }
    return false;
}

/// \brief Runs the restricted unpickler. \param bytes The pickle. \param limits The bounds.
/// \return The object graph, or the first refusal.
[[nodiscard]] inline result<pickled> unpickle(std::span<const std::byte> bytes, const unpickle_limits& limits = {})
{
    pickled p;
    std::vector<std::size_t> stack, memo;
    std::size_t at = 0;
    const auto need = [&](std::size_t n) { return n <= bytes.size() - at; };
    const auto make = [&](object o) -> result<std::size_t> {
        if (p.objects.size() >= limits.max_objects)
            return fail(errc::too_large, "pickle: more than " + std::to_string(limits.max_objects) + " objects");
        p.objects.push_back(std::move(o));
        return ok(p.objects.size() - 1);
    };
    const auto push = [&](std::size_t i) -> result<void> {
        if (stack.size() >= limits.max_stack)
            return fail(errc::too_large, "pickle: the stack is deeper than " + std::to_string(limits.max_stack));
        stack.push_back(i);
        return ok();
    };
    const auto pop = [&]() -> result<std::size_t> {
        if (stack.empty() || p.objects[stack.back()].kind == kind::mark)
            return fail(errc::bad_header, "pickle: the stack is empty");
        const std::size_t i = stack.back();
        stack.pop_back();
        return ok(i);
    };
    // The items above the topmost MARK, and the MARK removed
    const auto pop_mark = [&]() -> result<std::vector<std::size_t>> {
        std::size_t m = stack.size();
        while (m > 0 && p.objects[stack[m - 1]].kind != kind::mark)
            --m;
        if (m == 0)
            return fail(errc::bad_header, "pickle: no MARK on the stack");
        std::vector<std::size_t> items(stack.begin() + static_cast<std::ptrdiff_t>(m), stack.end());
        stack.resize(m - 1);
        return ok(std::move(items));
    };
    const auto memo_put = [&](std::uint64_t idx) -> result<void> {
        if (stack.empty())
            return fail(errc::bad_header, "pickle: PUT on an empty stack");
        if (idx >= limits.max_memo)
            return fail(errc::too_large, "pickle: memo index " + std::to_string(idx));
        if (memo.size() <= idx)
            memo.resize(static_cast<std::size_t>(idx) + 1, static_cast<std::size_t>(-1));
        memo[static_cast<std::size_t>(idx)] = stack.back();
        return ok();
    };
    const auto memo_get = [&](std::uint64_t idx) -> result<void> {
        if (idx >= memo.size() || memo[static_cast<std::size_t>(idx)] == static_cast<std::size_t>(-1))
            return fail(errc::bad_header, "pickle: GET of memo " + std::to_string(idx) + ", never PUT");
        return push(memo[static_cast<std::size_t>(idx)]);
    };
    const auto text = [&](std::size_t n, kind k) -> result<void> {
        if (!need(n))
            return fail(errc::truncated, "pickle: a string runs past the end");
        object o;
        o.kind = k;
        o.text.assign(reinterpret_cast<const char*>(bytes.data() + at), n);
        at += n;
        SNICHOLLS_TRY_ASSIGN(const std::size_t i, make(std::move(o)));
        return push(i);
    };
    const auto integer = [&](std::int64_t v) -> result<void> {
        object o;
        o.kind = kind::integer;
        o.integer = v;
        SNICHOLLS_TRY_ASSIGN(const std::size_t i, make(std::move(o)));
        return push(i);
    };
    const auto container = [&](kind k, std::vector<std::size_t> items) -> result<void> {
        object o;
        o.kind = k;
        o.items = std::move(items);
        SNICHOLLS_TRY_ASSIGN(const std::size_t i, make(std::move(o)));
        return push(i);
    };
    const auto global = [&](std::string_view module, std::string_view name) -> result<void> {
        if (!allowed_global(module, name))
            return fail(errc::unsafe_pickle, "pickle: global " + quote_name(std::string(module) + "." + std::string(name)) + " is not allowed");
        object o;
        o.kind = kind::global;
        o.text = std::string(module) + "." + std::string(name);
        SNICHOLLS_TRY_ASSIGN(const std::size_t i, make(std::move(o)));
        return push(i);
    };
    const auto line = [&]() -> std::optional<std::string_view> {
        const std::string_view rest(reinterpret_cast<const char*>(bytes.data() + at), bytes.size() - at);
        const auto nl = rest.find('\n');
        if (nl == std::string_view::npos || nl > 256)
            return std::nullopt;
        at += nl + 1;
        return rest.substr(0, nl);
    };
    while (true) {
        if (!need(1))
            return fail(errc::truncated, "pickle: no STOP");
        const auto op = std::to_integer<std::uint8_t>(bytes[at++]);
        switch (op) {
        case 0x80:   // PROTO
            if (!need(1))
                return fail(errc::truncated, "pickle: PROTO without a version");
            if (const auto v = std::to_integer<int>(bytes[at++]); v < 2 || v > 5)
                return fail(errc::unsafe_pickle, "pickle: protocol " + std::to_string(v) + " is not read");
            break;
        case 0x95:   // FRAME: the frame's length; the opcodes follow inline
            if (!need(8))
                return fail(errc::truncated, "pickle: FRAME without a length");
            at += 8;
            break;
        case '.': {  // STOP
            SNICHOLLS_TRY_ASSIGN(p.root, pop());
            return ok(std::move(p));
        }
        case '(': {  // MARK
            object m;
            m.kind = kind::mark;
            SNICHOLLS_TRY_ASSIGN(const std::size_t i, make(std::move(m)));
            SNICHOLLS_TRY_VOID(push(i));
            break;
        }
        case 'p': {  // PUT (text index)
            const auto l = line();
            std::uint64_t idx = 0;
            if (!l || std::from_chars(l->data(), l->data() + l->size(), idx).ec != std::errc{})
                return fail(errc::bad_header, "pickle: a malformed PUT");
            SNICHOLLS_TRY_VOID(memo_put(idx));
            break;
        }
        case 'q':    // BINPUT
            if (!need(1))
                return fail(errc::truncated, "pickle: BINPUT");
            SNICHOLLS_TRY_VOID(memo_put(std::to_integer<std::uint8_t>(bytes[at++])));
            break;
        case 'r':    // LONG_BINPUT
            if (!need(4))
                return fail(errc::truncated, "pickle: LONG_BINPUT");
            SNICHOLLS_TRY_VOID(memo_put(load_le<std::uint32_t>(bytes.data() + at)));
            at += 4;
            break;
        case 0x94:   // MEMOIZE
            SNICHOLLS_TRY_VOID(memo_put(memo.size()));
            break;
        case 'g': {  // GET (text index)
            const auto l = line();
            std::uint64_t idx = 0;
            if (!l || std::from_chars(l->data(), l->data() + l->size(), idx).ec != std::errc{})
                return fail(errc::bad_header, "pickle: a malformed GET");
            SNICHOLLS_TRY_VOID(memo_get(idx));
            break;
        }
        case 'h':    // BINGET
            if (!need(1))
                return fail(errc::truncated, "pickle: BINGET");
            SNICHOLLS_TRY_VOID(memo_get(std::to_integer<std::uint8_t>(bytes[at++])));
            break;
        case 'j':    // LONG_BINGET
            if (!need(4))
                return fail(errc::truncated, "pickle: LONG_BINGET");
            SNICHOLLS_TRY_VOID(memo_get(load_le<std::uint32_t>(bytes.data() + at)));
            at += 4;
            break;
        case 'N': {
            SNICHOLLS_TRY_ASSIGN(const std::size_t i, make(object{}));
            SNICHOLLS_TRY_VOID(push(i));
            break;
        }
        case 0x88:   // NEWTRUE
        case 0x89: { // NEWFALSE
            object o;
            o.kind = kind::boolean;
            o.integer = op == 0x88 ? 1 : 0;
            SNICHOLLS_TRY_ASSIGN(const std::size_t i, make(std::move(o)));
            SNICHOLLS_TRY_VOID(push(i));
            break;
        }
        case 'J':    // BININT
            if (!need(4))
                return fail(errc::truncated, "pickle: BININT");
            SNICHOLLS_TRY_VOID(integer(static_cast<std::int32_t>(load_le<std::uint32_t>(bytes.data() + at))));
            at += 4;
            break;
        case 'K':    // BININT1
            if (!need(1))
                return fail(errc::truncated, "pickle: BININT1");
            SNICHOLLS_TRY_VOID(integer(std::to_integer<std::uint8_t>(bytes[at++])));
            break;
        case 'M':    // BININT2
            if (!need(2))
                return fail(errc::truncated, "pickle: BININT2");
            SNICHOLLS_TRY_VOID(integer(load_le<std::uint16_t>(bytes.data() + at)));
            at += 2;
            break;
        case 0x8a: { // LONG1: little-endian two's complement, at most 8 bytes here
            if (!need(1))
                return fail(errc::truncated, "pickle: LONG1");
            const std::size_t n = std::to_integer<std::uint8_t>(bytes[at++]);
            if (n > 8 || !need(n))
                return fail(n > 8 ? errc::unsupported : errc::truncated, "pickle: a LONG1 of " + std::to_string(n) + " bytes");
            std::uint64_t v = 0;
            for (std::size_t k = 0; k < n; ++k)
                v |= std::uint64_t{std::to_integer<std::uint8_t>(bytes[at + k])} << (8 * k);
            if (n > 0 && n < 8 && (std::to_integer<std::uint8_t>(bytes[at + n - 1]) & 0x80))
                v |= ~std::uint64_t{0} << (8 * n);   // sign-extend
            at += n;
            SNICHOLLS_TRY_VOID(integer(static_cast<std::int64_t>(v)));
            break;
        }
        case 'G': {  // BINFLOAT: big-endian
            if (!need(8))
                return fail(errc::truncated, "pickle: BINFLOAT");
            object o;
            o.kind = kind::floating;
            o.floating = std::bit_cast<double>(load_be<std::uint64_t>(bytes.data() + at));
            at += 8;
            SNICHOLLS_TRY_ASSIGN(const std::size_t i, make(std::move(o)));
            SNICHOLLS_TRY_VOID(push(i));
            break;
        }
        case 'X':    // BINUNICODE
        case 'B':    // BINBYTES
            if (!need(4))
                return fail(errc::truncated, "pickle: a string length");
            at += 4;
            SNICHOLLS_TRY_VOID(text(load_le<std::uint32_t>(bytes.data() + at - 4), op == 'X' ? kind::string : kind::bytes));
            break;
        case 0x8c:   // SHORT_BINUNICODE
        case 'C':    // SHORT_BINBYTES
            if (!need(1))
                return fail(errc::truncated, "pickle: a string length");
            ++at;
            SNICHOLLS_TRY_VOID(text(std::to_integer<std::uint8_t>(bytes[at - 1]), op == 'C' ? kind::bytes : kind::string));
            break;
        case 0x8d: { // BINUNICODE8
            if (!need(8))
                return fail(errc::truncated, "pickle: a string length");
            const std::uint64_t n = load_le<std::uint64_t>(bytes.data() + at);
            at += 8;
            if (n > bytes.size())
                return fail(errc::truncated, "pickle: a string runs past the end");
            SNICHOLLS_TRY_VOID(text(static_cast<std::size_t>(n), kind::string));
            break;
        }
        case ')':
            SNICHOLLS_TRY_VOID(container(kind::tuple, {}));
            break;
        case ']':
            SNICHOLLS_TRY_VOID(container(kind::list, {}));
            break;
        case '}':
            SNICHOLLS_TRY_VOID(container(kind::dict, {}));
            break;
        case 't': {
            SNICHOLLS_TRY_ASSIGN(std::vector<std::size_t> items, pop_mark());
            SNICHOLLS_TRY_VOID(container(kind::tuple, std::move(items)));
            break;
        }
        case 0x85:
        case 0x86:
        case 0x87: {
            const std::size_t n = op - 0x84u;
            std::vector<std::size_t> items(n);
            for (std::size_t k = n; k > 0; --k) {
                SNICHOLLS_TRY_ASSIGN(items[k - 1], pop());
            }
            SNICHOLLS_TRY_VOID(container(kind::tuple, std::move(items)));
            break;
        }
        case 'a': {  // APPEND
            SNICHOLLS_TRY_ASSIGN(const std::size_t v, pop());
            if (stack.empty() || p.objects[stack.back()].kind != kind::list)
                return fail(errc::bad_header, "pickle: APPEND to something not a list");
            p.objects[stack.back()].items.push_back(v);
            break;
        }
        case 'e': {  // APPENDS
            SNICHOLLS_TRY_ASSIGN(std::vector<std::size_t> items, pop_mark());
            if (stack.empty() || p.objects[stack.back()].kind != kind::list)
                return fail(errc::bad_header, "pickle: APPENDS to something not a list");
            auto& l = p.objects[stack.back()].items;
            l.insert(l.end(), items.begin(), items.end());
            break;
        }
        case 's': {  // SETITEM
            SNICHOLLS_TRY_ASSIGN(const std::size_t v, pop());
            SNICHOLLS_TRY_ASSIGN(const std::size_t k, pop());
            if (stack.empty() || p.objects[stack.back()].kind != kind::dict)
                return fail(errc::bad_header, "pickle: SETITEM on something not a dict");
            p.objects[stack.back()].items.push_back(k);
            p.objects[stack.back()].items.push_back(v);
            break;
        }
        case 'u': {  // SETITEMS
            SNICHOLLS_TRY_ASSIGN(std::vector<std::size_t> items, pop_mark());
            if (items.size() % 2 != 0 || stack.empty() || p.objects[stack.back()].kind != kind::dict)
                return fail(errc::bad_header, "pickle: SETITEMS on something not a dict, or an odd count");
            auto& d = p.objects[stack.back()].items;
            d.insert(d.end(), items.begin(), items.end());
            break;
        }
        case 'c': {  // GLOBAL: module\nname\n
            const auto module = line();
            const auto name = module ? line() : std::nullopt;
            if (!module || !name)
                return fail(errc::truncated, "pickle: a malformed GLOBAL");
            SNICHOLLS_TRY_VOID(global(*module, *name));
            break;
        }
        case 0x93: { // STACK_GLOBAL
            SNICHOLLS_TRY_ASSIGN(const std::size_t n, pop());
            SNICHOLLS_TRY_ASSIGN(const std::size_t m, pop());
            if (p.objects[m].kind != kind::string || p.objects[n].kind != kind::string)
                return fail(errc::bad_header, "pickle: STACK_GLOBAL of something not strings");
            const std::string module = p.objects[m].text, name = p.objects[n].text;
            SNICHOLLS_TRY_VOID(global(module, name));
            break;
        }
        case 'Q': {  // BINPERSID: ('storage', type, key, location, numel)
            SNICHOLLS_TRY_ASSIGN(const std::size_t pid, pop());
            const object& t = p.objects[pid];
            const auto is = [&](std::size_t k, kind want) { return t.items.size() > k && p.objects[t.items[k]].kind == want; };
            if (t.kind != kind::tuple || t.items.size() != 5 || !is(0, kind::string) || p.objects[t.items[0]].text != "storage"
                || !is(1, kind::global) || (!is(2, kind::string) && !is(2, kind::integer)) || !is(4, kind::integer))
                return fail(errc::bad_header, "pickle: a persistent id that is not ('storage', type, key, location, numel)");
            object s;
            s.kind = kind::storage;
            const std::string& type = p.objects[t.items[1]].text;
            s.text = type.substr(type.rfind('.') + 1);
            s.key = p.objects[t.items[2]].kind == kind::string ? p.objects[t.items[2]].text : std::to_string(p.objects[t.items[2]].integer);
            s.integer = p.objects[t.items[4]].integer;
            if (s.text.find("Storage") == std::string::npos || s.integer < 0)
                return fail(errc::bad_header, "pickle: a persistent id whose type is not a storage");
            SNICHOLLS_TRY_ASSIGN(const std::size_t i, make(std::move(s)));
            SNICHOLLS_TRY_VOID(push(i));
            break;
        }
        case 'R': {  // REDUCE: callable(*args), for the allowed callables only
            SNICHOLLS_TRY_ASSIGN(const std::size_t args, pop());
            SNICHOLLS_TRY_ASSIGN(const std::size_t fn, pop());
            const object& f = p.objects[fn];
            const object& a = p.objects[args];
            if (f.kind != kind::global || a.kind != kind::tuple)
                return fail(errc::unsafe_pickle, "pickle: REDUCE of something not an allowed global");
            if (f.text == "collections.OrderedDict") {
                if (!a.items.empty())
                    return fail(errc::unsafe_pickle, "pickle: OrderedDict with arguments");
                SNICHOLLS_TRY_VOID(container(kind::dict, {}));
            } else if (f.text == "torch._utils._rebuild_tensor_v2") {
                const auto arg = [&](std::size_t k) -> const object& { return p.objects[a.items[k]]; };
                if (a.items.size() < 4 || arg(0).kind != kind::storage || arg(1).kind != kind::integer || arg(2).kind != kind::tuple
                    || arg(3).kind != kind::tuple)
                    return fail(errc::bad_header, "pickle: _rebuild_tensor_v2 without (storage, offset, size, stride)");
                SNICHOLLS_TRY_VOID(container(kind::tensor, {a.items[0], a.items[1], a.items[2], a.items[3]}));
            } else if (f.text == "torch._utils._rebuild_parameter") {
                if (a.items.empty() || p.objects[a.items[0]].kind != kind::tensor)
                    return fail(errc::bad_header, "pickle: _rebuild_parameter without a tensor");
                SNICHOLLS_TRY_VOID(push(a.items[0]));
            } else {
                return fail(errc::unsafe_pickle, "pickle: REDUCE of " + quote_name(f.text) + " is not allowed");
            }
            break;
        }
        case 'b': {  // BUILD: state on a dict or tensor is discarded
            SNICHOLLS_TRY_ASSIGN(const std::size_t state, pop());
            (void)state;
            if (stack.empty() || (p.objects[stack.back()].kind != kind::dict && p.objects[stack.back()].kind != kind::tensor))
                return fail(errc::unsafe_pickle, "pickle: BUILD on something not a dict or tensor");
            break;
        }
        default: {
            char hex[8];
            std::snprintf(hex, sizeof hex, "0x%02x", op);
            return fail(errc::unsafe_pickle, std::string("pickle: opcode ") + hex + " at byte " + std::to_string(at - 1) + " is not allowed");
        }
        }
    }
}
} // namespace snicholls::formats::pytorch

#endif // snicholls_formats_pytorch_pickle_hpp
