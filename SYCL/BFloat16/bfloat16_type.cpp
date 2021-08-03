// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

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

template <typename T> void assert_close(const T &C, const float ref) {
  for (size_t i = 0; i < N; i++) {
    auto diff = C[i] - ref;
    assert(std::fabs(static_cast<float>(diff)) <
           std::numeric_limits<float>::epsilon());
  }
}

void verify_conv(queue &q, buffer<float, 1> &a, range<1> &r, const float ref) {
  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class calc_conv>(r, [=](id<1> index) {
      cl::sycl::ext::intel::experimental::bfloat16 AVal{A[index]};
      A[index] = AVal;
    });
  });

  assert_close(a.get_access<access::mode::read>(), ref);
}

void verify_add(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_add>(r, [=](id<1> index) {
      cl::sycl::ext::intel::experimental::bfloat16 AVal{A[index]};
      cl::sycl::ext::intel::experimental::bfloat16 BVal{B[index]};
      cl::sycl::ext::intel::experimental::bfloat16 CVal =
          static_cast<float>(AVal) + static_cast<float>(BVal);
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_min(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_min>(r, [=](id<1> index) {
      cl::sycl::ext::intel::experimental::bfloat16 AVal{A[index]};
      cl::sycl::ext::intel::experimental::bfloat16 BVal{B[index]};
      cl::sycl::ext::intel::experimental::bfloat16 CVal =
          static_cast<float>(AVal) - static_cast<float>(BVal);
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_mul(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_mul>(r, [=](id<1> index) {
      cl::sycl::ext::intel::experimental::bfloat16 AVal{A[index]};
      cl::sycl::ext::intel::experimental::bfloat16 BVal{B[index]};
      cl::sycl::ext::intel::experimental::bfloat16 CVal =
          static_cast<float>(AVal) * static_cast<float>(BVal);
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

void verify_div(queue &q, buffer<float, 1> &a, buffer<float, 1> &b, range<1> &r,
                const float ref) {
  buffer<float, 1> c{r};

  q.submit([&](handler &cgh) {
    auto A = a.get_access<access::mode::read>(cgh);
    auto B = b.get_access<access::mode::read>(cgh);
    auto C = c.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class calc_div>(r, [=](id<1> index) {
      cl::sycl::ext::intel::experimental::bfloat16 AVal{A[index]};
      cl::sycl::ext::intel::experimental::bfloat16 BVal{B[index]};
      cl::sycl::ext::intel::experimental::bfloat16 CVal =
          static_cast<float>(AVal) / static_cast<float>(BVal);
      C[index] = CVal;
    });
  });

  assert_close(c.get_access<access::mode::read>(), ref);
}

int main() {
  device dev{default_selector()};

  // TODO: replace is_gpu check with extension check when the appropriate part
  // of implementation ready
  if (!dev.is_gpu()) {
    std::cout << "This device doesn't support bfloat16 type" << std::endl;
    return 0;
  }

  std::vector<float> vec_a(N, 5.0);
  std::vector<float> vec_b(N, 2.0);

  range<1> r(N);
  buffer<float, 1> a{vec_a.data(), r};
  buffer<float, 1> b{vec_b.data(), r};

  queue q{dev};

  verify_conv(q, a, r, 5.0);
  verify_add(q, a, b, r, 7.0);
  verify_min(q, a, b, r, 3.0);
  verify_mul(q, a, b, r, 10.0);
  verify_div(q, a, b, r, 2.5);

  return 0;
}
