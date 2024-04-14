---
title: "MLOps CICD"
layout: archive
permalink: /categories/MLOps-cicd
author_profile: true
types: posts
---

{% assign posts = site.categories["MLOps-cicd"]%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}