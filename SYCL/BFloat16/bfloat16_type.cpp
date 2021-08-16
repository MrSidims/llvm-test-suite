// RUN: %clangxx -fsycl -fsycl-enable-bfloat16-conversion -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t2.out
// RUN: %CPU_RUN_PLACEHOLDER %t2.out

//==----------- bfloat16_type.cpp - SYCL bfloat16 type test ----------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>
#include <sycl/ext/intel/experimental/bfloat16.hpp>

#include <cmath>

using namespace cl::sycl;

constexpr size_t N = 100;

uint16_t lib_to_float(float Val) {
#ifdef __SYCL_ENABLE_BF16_CONVERSION__
  return cl::sycl::ext::intel::experimental::bfloat16::from_float(Val);
#else
  // That is not a correct convertion, just implement it this way to simplify
  // checks on the host
  return (uint16_t)(Val);
#endif // __SYCL_ENABLE_BF16_CONVERSION__
}

float lib_from_float(uint16_t Val) {
#ifdef __SYCL_ENABLE_BF16_CONVERSION__
  return cl::sycl::ext::intel::experimental::bfloat16::to_float(Val);
#else
  // That is not a correct convertion, just implement it this way to simplify
  // checks on the host
  return (float)(Val);
#endif // __SYCL_ENABLE_BF16_CONVERSION__
}


void verify_conv_lib(queue &q, buffer<float, 1> &a, range<1> &r,
                     const float ref) {
  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class calc_conv_impl>(r, [=](id<1> index) {
      uint16_t AVal = lib_from_float(A[index]);
      A[index] = lib_to_float(AVal);
    });
  });

  assert_close(a.get_access<access::mode::read>(), ref);
}

template <typename T> void assert_close(const T &C, const float ref) {
  for (size_t i = 0; i < N; i++) {
    auto diff = C[i] - ref;
    assert(std::fabs(static_cast<float>(diff)) <
           std::numeric_limits<float>::epsilon());
  }
}

void verify_conv_implicit(queue &q, buffer<float, 1> &a, range<1> &r,
                          const float ref) {
#ifdef __SYCL_ENABLE_BF16_CONVERSION__
  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class calc_conv>(r, [=](id<1> index) {
      cl::sycl::ext::intel::experimental::bfloat16 AVal{A[index]};
      A[index] = AVal;
    });
  });

  assert_close(a.get_access<access::mode::read>(), ref);
#endif // __SYCL_ENABLE_BF16_CONVERSION__
}

void verify_conv_explicit(queue &q, buffer<float, 1> &a, range<1> &r,
                          const float ref) {
#ifdef __SYCL_ENABLE_BF16_CONVERSION__
  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class calc_conv_impl>(r, [=](id<1> index) {
      uint16_t AVal =
          cl::sycl::ext::intel::experimental::bfloat16::from_float(A[index]);
      A[index] = cl::sycl::ext::intel::experimental::bfloat16::to_float(AVal);
    });
  });

  assert_close(a.get_access<access::mode::read>(), ref);
#endif // __SYCL_ENABLE_BF16_CONVERSION__
}

void verify_add(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
#ifdef __SYCL_ENABLE_BF16_CONVERSION__
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_add_expl>(r, [=](id<1> index) {
      cl::sycl::ext::intel::experimental::bfloat16 AVal{A[index]};
      cl::sycl::ext::intel::experimental::bfloat16 BVal{B[index]};
      cl::sycl::ext::intel::experimental::bfloat16 CVal = AVal + BVal;
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
#endif // __SYCL_ENABLE_BF16_CONVERSION__
}

void verify_sub(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
#ifdef __SYCL_ENABLE_BF16_CONVERSION__
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_sub>(r, [=](id<1> index) {
      cl::sycl::ext::intel::experimental::bfloat16 AVal{A[index]};
      cl::sycl::ext::intel::experimental::bfloat16 BVal{B[index]};
      cl::sycl::ext::intel::experimental::bfloat16 CVal = AVal - BVal;
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
#endif // __SYCL_ENABLE_BF16_CONVERSION__
}

void verify_mul(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
#ifdef __SYCL_ENABLE_BF16_CONVERSION__
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_mul>(r, [=](id<1> index) {
      cl::sycl::ext::intel::experimental::bfloat16 AVal{A[index]};
      cl::sycl::ext::intel::experimental::bfloat16 BVal{B[index]};
      cl::sycl::ext::intel::experimental::bfloat16 CVal = AVal * BVal;
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
#endif // __SYCL_ENABLE_BF16_CONVERSION__
}

void verify_div(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
#ifdef __SYCL_ENABLE_BF16_CONVERSION__
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_div>(r, [=](id<1> index) {
      cl::sycl::ext::intel::experimental::bfloat16 AVal{A[index]};
      cl::sycl::ext::intel::experimental::bfloat16 BVal{B[index]};
      cl::sycl::ext::intel::experimental::bfloat16 CVal = AVal / BVal;
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
#endif // __SYCL_ENABLE_BF16_CONVERSION__
}

void verify_logic(queue &q, buffer<float, 1> &a, buffer<float, 1> &b,
                  range<1> &r, const float ref) {
#ifdef __SYCL_ENABLE_BF16_CONVERSION__
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class logic>(r, [=](id<1> index) {
      cl::sycl::ext::intel::experimental::bfloat16 AVal{A[index]};
      cl::sycl::ext::intel::experimental::bfloat16 BVal{B[index]};
      if (AVal && !BVal) {
        if (AVal > BVal || AVal >= BVal || AVal < BVal || AVal <= BVal) {
          cl::sycl::ext::intel::experimental::bfloat16 CVal =
              AVal != BVal ? AVal : BVal;
          CVal--;
          CVal++;
          if (AVal == BVal) {
            CVal -= AVal;
            CVal *= 3.0;
            CVal /= 2.0;
          } else
            CVal += BVal;
        }
      }
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
#endif // __SYCL_ENABLE_BF16_CONVERSION__
}

int main() {
  device dev{default_selector()};

  // TODO: replace is_gpu check with extension check when the appropriate part
  // of implementation ready (aspect)
#ifdef __SYCL_ENABLE_BF16_CONVERSION__
  if (!dev.is_gpu()) {
    std::cout << "This device doesn't support bfloat16 conversion feature"
              << std::endl;
    return 0;
  }
#endif // __SYCL_ENABLE_BF16_CONVERSION__

  std::vector<float> vec_a(N, 5.0);
  std::vector<float> vec_b(N, 2.0);
  std::vector<float> vec_b_neg(N, -2.0);

  range<1> r(N);
  buffer<float, 1> a{vec_a.data(), r};
  buffer<float, 1> b{vec_b.data(), r};
  buffer<float, 1> b_neg{vec_b_neg.data(), r};

  queue q{dev};

  verify_conv_implicit(q, a, r, 5.0);
  verify_conv_explicit(q, a, r, 5.0);
  verify_conv_lib(q, a, r, 5.0);
  verify_add(q, a, b, r, 7.0);
  verify_sub(q, a, b, r, 3.0);
  verify_mul(q, a, b, r, 10.0);
  verify_div(q, a, b, r, 2.5);
  verify_logic(q, a, b, r, 7.0);
  verify_add(q, a, b_neg, r, 3.0);
  verify_sub(q, a, b_neg, r, 7.0);
  verify_mul(q, a, b_neg, r, -10.0);
  verify_div(q, a, b_neg, r, -2.5);
  verify_logic(q, a, b_neg, r, 3.0);

  return 0;
}
