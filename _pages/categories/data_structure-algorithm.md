---
title: "Data Structure & Algorithm"
layout: archive
permalink: /categories/data_structure-algorithm
author_profile: true
types: posts
---

{% assign posts = site.categories["data_structure-algorithm"]%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}