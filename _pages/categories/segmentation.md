---
title: "Segmentation"
layout: archive
permalink: /categories/segmentation
author_profile: true
types: posts
---

{% assign posts = site.categories["segmentation"]%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}
