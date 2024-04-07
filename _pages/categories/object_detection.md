---
title: "Object Detection"
layout: archive
permalink: /categories/object_detection
author_profile: true
types: posts
---

{% assign posts = site.categories["object_detection"]%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}