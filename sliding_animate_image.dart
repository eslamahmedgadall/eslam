import 'package:flutter/material.dart';
import 'package:splash_project/utils/assets.dart';

class slidingAinmateImage extends StatelessWidget {
  const slidingAinmateImage({
    super.key,
    required this.slidingAnimation,
  });

  final Animation<Offset> slidingAnimation;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
        animation: slidingAnimation,
        builder: (context, _) {
          return SlideTransition(
            position: slidingAnimation,
            child: Image.asset(AssetsData.logo),
          );
        });
  }
}
