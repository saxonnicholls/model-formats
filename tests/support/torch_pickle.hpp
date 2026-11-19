//
//  torch_pickle.hpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//
//
//  Test support: pickles assembled opcode by opcode as torch.save (protocol 2) writes a state dict, and the zip around
//  them, so the restricted unpickler and the checkpoint reader are tested without PyTorch.

#ifndef model_formats_tests_support_torch_pickle_hpp
#define model_formats_tests_support_torch_pickle_hpp

#include <snicholls/formats/archive/zip.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace test_support
{
struct pickle_writer
{
    std::vector<std::byte> out;
    pickle_writer& op(std::uint8_t c)
    {
        out.push_back(std::byte{c});
        return *this;
    }
    pickle_writer& raw(std::string_view s)
    {
        for (const char c : s)
            out.push_back(static_cast<std::byte>(c));
        return *this;
    }
    pickle_writer& str(std::string_view s)   // BINUNICODE
    {
        op('X');
        const auto n = static_cast<std::uint32_t>(s.size());
        for (int k = 0; k < 4; ++k)
            op(static_cast<std::uint8_t>(n >> (8 * k)));
        return raw(s);
    }
    pickle_writer& global(std::string_view module, std::string_view name) { return op('c').raw(module).raw("\n").raw(name).raw("\n"); }
    pickle_writer& integer(std::uint8_t v) { return op('K').op(v); }
    // _rebuild_tensor_v2(storage, offset, size, stride, False, OrderedDict()) with the storage as a persistent id
    pickle_writer& tensor(std::string_view storage_type, std::string_view key, std::uint8_t numel, std::uint8_t offset,
                          std::vector<std::uint8_t> size, std::vector<std::uint8_t> stride)
    {
        global("torch._utils", "_rebuild_tensor_v2");
        op('(');
        op('(').str("storage").global("torch", storage_type).str(key).str("cpu").integer(numel).op('t').op('Q');
        integer(offset);
        op('(');
        for (const auto s : size)
            integer(s);
        op('t');
        op('(');
        for (const auto s : stride)
            integer(s);
        op('t');
        op(0x89);   // requires_grad False
        global("collections", "OrderedDict").op(')').op('R');
        return op('t').op('R');
    }
};

// A torch.save zip: <prefix>/data.pkl, <prefix>/byteorder, and <prefix>/data/<key> for each storage
inline std::vector<std::byte> torch_zip(const std::vector<std::byte>& pkl, const std::vector<std::pair<std::string, std::vector<std::byte>>>& storages,
                                        std::string prefix = "archive")
{
    std::vector<std::string> names{prefix + "/data.pkl", prefix + "/byteorder"};
    for (const auto& [k, b] : storages)
        names.push_back(prefix + "/data/" + k);
    static const std::string little = "little";
    std::vector<snicholls::formats::zip::member> members{{names[0], pkl}, {names[1], std::as_bytes(std::span(little.data(), little.size()))}};
    for (std::size_t i = 0; i < storages.size(); ++i)
        members.push_back({names[2 + i], storages[i].second});
    return *snicholls::formats::zip::write(members);
}
} // namespace test_support

#endif // model_formats_tests_support_torch_pickle_hpp
