#include "CLUEstering/CLUEstering.hpp"

#include <algorithm>
#include <span>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

TEST_CASE("CylinderMetric: distance values") {
  auto queue = clue::get_queue(0u);
  clue::PointsHost<3> points(queue, 2);
  const std::vector<float> x = {0.f, 0.9f};
  const std::vector<float> y = {0.f, 1.2f};
  const std::vector<float> z = {0.f, 0.5f};
  std::ranges::copy(x, points.coords(0).begin());
  std::ranges::copy(y, points.coords(1).begin());
  std::ranges::copy(z, points.coords(2).begin());

  auto metric = clue::metrics::Cylinder<3>{};

  SUBCASE("Without sigma: max(transverse Euclidean, axial)") {
    CHECK(metric(points.view(), 0, 1) == doctest::Approx(1.5f));
  }

  SUBCASE("Transverse sigma divides the transverse distance by the pair mean") {
    const std::vector<float> sigma = {1.f, 3.f};
    points.set_sigma(0, std::span(sigma));
    points.set_sigma(1, std::span(sigma));
    CHECK(metric(points.view(), 0, 1) == doctest::Approx(0.75f));
  }

  SUBCASE("Axial sigma is ignored") {
    const std::vector<float> sigma = {2.f, 2.f};
    points.set_sigma(2, std::span(sigma));
    CHECK(metric(points.view(), 0, 1) == doctest::Approx(1.5f));
  }
}

TEST_CASE("CylinderMetric: sigma controls cluster merging") {
  auto queue = clue::get_queue(0u);

  const float density_radius = 1.f;
  const float min_density = 0.3f;
  clue::Clusterer<3> algo(queue, density_radius, min_density);

  const std::vector<float> weights = {1.f, 1.f};

  SUBCASE("Transverse pair at distance 1.5 without sigma => 2 clusters") {
    clue::PointsHost<3> points(queue, 2);
    const std::vector<float> x = {0.f, 0.9f};
    const std::vector<float> y = {0.f, 1.2f};
    const std::vector<float> z = {0.f, 0.f};
    std::ranges::copy(x, points.coords(0).begin());
    std::ranges::copy(y, points.coords(1).begin());
    std::ranges::copy(z, points.coords(2).begin());
    std::ranges::copy(weights, points.weights().begin());

    algo.make_clusters(queue, points, clue::metrics::Cylinder<3>{});

    CHECK(points.n_clusters() == 2);
  }

  SUBCASE("Transverse pair at distance 1.5 with sigma 2 => 1 cluster") {
    clue::PointsHost<3> points(queue, 2);
    const std::vector<float> x = {0.f, 0.9f};
    const std::vector<float> y = {0.f, 1.2f};
    const std::vector<float> z = {0.f, 0.f};
    std::ranges::copy(x, points.coords(0).begin());
    std::ranges::copy(y, points.coords(1).begin());
    std::ranges::copy(z, points.coords(2).begin());
    std::ranges::copy(weights, points.weights().begin());

    const std::vector<float> sigma = {2.f, 2.f};
    points.set_sigma(0, std::span(sigma));
    points.set_sigma(1, std::span(sigma));

    algo.make_clusters(queue, points, clue::metrics::Cylinder<3>{});

    CHECK(points.n_clusters() == 1);
  }

  SUBCASE("Axial pair at distance 1.5 with sigma 2 on every dimension => still 2 clusters") {
    clue::PointsHost<3> points(queue, 2);
    const std::vector<float> x = {0.f, 0.f};
    const std::vector<float> y = {0.f, 0.f};
    const std::vector<float> z = {0.f, 1.5f};
    std::ranges::copy(x, points.coords(0).begin());
    std::ranges::copy(y, points.coords(1).begin());
    std::ranges::copy(z, points.coords(2).begin());
    std::ranges::copy(weights, points.weights().begin());

    const std::vector<float> sigma = {2.f, 2.f};
    points.set_sigmas(sigma, sigma, sigma);

    algo.make_clusters(queue, points, clue::metrics::Cylinder<3>{});

    CHECK(points.n_clusters() == 2);
  }
}
