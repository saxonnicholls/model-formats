//
//  pickle_test.cpp
//  model-formats
//
//  Copyright 2010-2026 Saxon Herschel Nicholls
//

#include <snicholls/formats/pytorch/pickle.hpp>

#include <gtest/gtest.h>

#include "support/torch_pickle.hpp"

using namespace snicholls::formats;
using namespace snicholls::formats::pytorch;
using test_support::pickle_writer;

namespace
{
errc code_of(const pickle_writer& w) { return unpickle(w.out).unwrap_err_unchecked().code; }
} // namespace

TEST(PytorchPickle, ReadsAStateDict)
{
    pickle_writer w;
    w.op(0x80).op(2).op('}').op('q').op(0).op('(').str("w").tensor("FloatStorage", "0", 6, 0, {2, 3}, {3, 1});
    w.str("n").integer(7).op('u').op('.');
    const auto p = unpickle(w.out);
    ASSERT_TRUE(p.is_ok()) << p.unwrap_err_unchecked().what();
    const object& root = p->at(p->root);
    ASSERT_EQ(root.kind, kind::dict);
    ASSERT_EQ(root.items.size(), 4u);
    const object& t = p->at(root.items[1]);
    ASSERT_EQ(t.kind, kind::tensor);
    const object& s = p->at(t.items[0]);
    EXPECT_EQ(s.kind, kind::storage);
    EXPECT_EQ(s.text, "FloatStorage");
    EXPECT_EQ(s.key, "0");
    EXPECT_EQ(s.integer, 6);
    EXPECT_EQ(p->at(root.items[3]).integer, 7);
}

TEST(PytorchPickle, MemoAndContainers)
{
    pickle_writer w;
    w.op(0x80).op(4).op(0x95);
    for (int i = 0; i < 8; ++i)
        w.op(0);
    w.op(']').op(0x94).op('(').integer(1).op(0x8a).op(2).op(0xff).op(0xff).op('e');   // [1, -1]
    w.op('h').op(0).op(0x86);                                                          // (list, list): the same list twice
    w.op(0x88).op(0x89).op('N').op(0x87).op(0x86);                                     // ((l, l), (True, False, None))
    w.op('.');
    const auto p = unpickle(w.out);
    ASSERT_TRUE(p.is_ok()) << p.unwrap_err_unchecked().what();
    const object& root = p->at(p->root);
    ASSERT_EQ(root.kind, kind::tuple);
    const object& pair = p->at(root.items[0]);
    EXPECT_EQ(pair.items[0], pair.items[1]);   // the memo returns the same object
    const object& list = p->at(pair.items[0]);
    ASSERT_EQ(list.items.size(), 2u);
    EXPECT_EQ(p->at(list.items[1]).integer, -1);
    EXPECT_EQ(p->at(p->at(root.items[1]).items[2]).kind, kind::none);
}

TEST(PytorchPickle, RefusesCodeExecution)
{
    pickle_writer sys;   // the classic: os.system("...")
    sys.op(0x80).op(2).global("os", "system").str("echo pwned").op(0x85).op('R').op('.');
    EXPECT_EQ(code_of(sys), errc::unsafe_pickle);
    pickle_writer eval;
    eval.op(0x80).op(2).global("builtins", "eval").str("1").op(0x85).op('R').op('.');
    EXPECT_EQ(code_of(eval), errc::unsafe_pickle);
    pickle_writer stack;   // STACK_GLOBAL of a forbidden name
    stack.op(0x80).op(4).str("subprocess").str("Popen").op(0x93).op('.');
    EXPECT_EQ(code_of(stack), errc::unsafe_pickle);
    pickle_writer storage_call;   // REDUCE of an allowed global that is not callable here
    storage_call.op(0x80).op(2).global("torch", "FloatStorage").op(')').op('R').op('.');
    EXPECT_EQ(code_of(storage_call), errc::unsafe_pickle);
    pickle_writer newobj;
    newobj.op(0x80).op(2).global("collections", "OrderedDict").op(')').op(0x81).op('.');
    EXPECT_EQ(code_of(newobj), errc::unsafe_pickle);
    pickle_writer inst;
    inst.op(0x80).op(2).op('(').raw("i").raw("os\nsystem\n").op('.');
    EXPECT_EQ(code_of(inst), errc::unsafe_pickle);
    pickle_writer build;   // BUILD on a list: __setstate__ is a call
    build.op(0x80).op(2).op(']').op('}').op('b').op('.');
    EXPECT_EQ(code_of(build), errc::unsafe_pickle);
    pickle_writer proto0;
    proto0.op(0x80).op(1).op('.');
    EXPECT_EQ(code_of(proto0), errc::unsafe_pickle);
    pickle_writer ext;
    ext.op(0x80).op(2).op(0x82).op(1).op('.');   // EXT1: the copyreg extension registry
    EXPECT_EQ(code_of(ext), errc::unsafe_pickle);
    EXPECT_TRUE(allowed_global("torch", "BFloat16Storage") && !allowed_global("torch", "load") && !allowed_global("torch._utils", "_rebuild_qtensor"));
}

TEST(PytorchPickle, RefusesMalformedPrograms)
{
    pickle_writer empty;
    empty.op(0x80).op(2).op('.');
    EXPECT_EQ(code_of(empty), errc::bad_header);
    pickle_writer no_stop;
    no_stop.op(0x80).op(2).op('N');
    EXPECT_EQ(code_of(no_stop), errc::truncated);
    pickle_writer bad_get;
    bad_get.op(0x80).op(2).op('h').op(5).op('.');
    EXPECT_EQ(code_of(bad_get), errc::bad_header);
    pickle_writer long_string;
    long_string.op(0x80).op(2).op('X').op(0xff).op(0xff).op(0xff).op(0x7f).op('.');
    EXPECT_EQ(code_of(long_string), errc::truncated);
    pickle_writer setitem;
    setitem.op(0x80).op(2).op(']').integer(1).integer(2).op('s').op('.');
    EXPECT_EQ(code_of(setitem), errc::bad_header);
    pickle_writer pid;
    pid.op(0x80).op(2).str("x").op('Q').op('.');
    EXPECT_EQ(code_of(pid), errc::bad_header);
    pickle_writer deep;
    deep.op(0x80).op(2);
    for (int i = 0; i < 100; ++i)
        deep.op(']');
    deep.op('.');
    EXPECT_EQ(unpickle(deep.out, unpickle_limits{50, 1u << 24, 1u << 20}).unwrap_err_unchecked().code, errc::too_large);
    EXPECT_EQ(unpickle(deep.out, unpickle_limits{1u << 24, 1u << 24, 10}).unwrap_err_unchecked().code, errc::too_large);
}
