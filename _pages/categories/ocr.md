---
title: "OCR"
layout: archive
permalink: /categories/ocr
author_profile: true
types: posts
---

{% assign posts = site.categories["ocr"]%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}